"""
GPU-side ImageNet JPEG decode+resize pipeline (nvJPEG via NVIDIA DALI), feeding
JAX arrays directly for train_rtx5090.py. See the DALI migration plan for context.

Smoke-tested against RTX 5090 (Blackwell/sm_120) in dali_smoke_test.py before
this module was written: nvJPEG hardware decode is confirmed working and pixel
values match torchvision's CPU decode within antialiasing tolerance. That test
also showed DALI's decoder silently tolerates truncated/corrupt JPEGs (no
exception) rather than failing loudly, which is why bad files are pre-filtered
here instead of relying on a runtime try/except.
"""
import glob
import hashlib
import json
import os

from nvidia.dali import pipeline_def, fn, types
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.jax import DALIGenericIterator


def build_file_label_lists(root_dir: str):
    """Same alphabetical class_to_idx logic as FastImageNetDataset, extracted so
    both the torch and DALI pipelines assign identical label indices."""
    dirs = sorted(d for d in glob.glob(os.path.join(root_dir, "n*")) if os.path.isdir(d))
    class_to_idx = {os.path.basename(d): i for i, d in enumerate(dirs)}

    paths, labels = [], []
    for d in dirs:
        idx = class_to_idx[os.path.basename(d)]
        for img in glob.glob(os.path.join(d, "*.JPEG")):
            paths.append(img)
            labels.append(idx)
    return paths, labels, class_to_idx


def filter_corrupt_files(paths, labels, cache_dir):
    """Drop unreadable/truncated JPEGs once, cache the filtered lists.

    DALI's mixed-device JPEG decoder does not raise on truncated/corrupt input
    (confirmed in dali_smoke_test.py) — it can silently produce partial or
    garbage image data instead. Unlike the torch pipeline's per-sample
    zero-tensor fallback, there is no cheap way to catch this at batch time, so
    bad files are excluded from the dataset up front via PIL's cheap verify().
    """
    from PIL import Image

    cache_key = hashlib.sha1(f"{len(paths)}:{paths[0] if paths else ''}".encode()).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f".dali_valid_files_cache_{cache_key}.json")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if cached.get("total_input") == len(paths):
            print(f"Using cached corrupt-file filter results: {len(cached['paths'])}/{len(paths)} valid.")
            return cached["paths"], cached["labels"]

    print(f"Scanning {len(paths)} images for corrupt/truncated JPEGs (one-time, cached after)...")
    valid_paths, valid_labels = [], []
    n_bad = 0
    for path, label in zip(paths, labels):
        try:
            with Image.open(path) as img:
                img.verify()
            valid_paths.append(path)
            valid_labels.append(label)
        except Exception:
            n_bad += 1

    print(f"Corrupt-file scan complete: {n_bad} bad file(s) excluded, {len(valid_paths)} valid images remain.")
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"total_input": len(paths), "paths": valid_paths, "labels": valid_labels}, f)

    return valid_paths, valid_labels


@pipeline_def
def imagenet_pipeline(files, labels, random_shuffle=True, shuffle_seed=0):
    jpegs, lbls = fn.readers.file(
        files=files, labels=labels,
        random_shuffle=random_shuffle, seed=shuffle_seed,
        name="Reader",
    )
    images = fn.decoders.image(
        jpegs, device="mixed", output_type=types.RGB,
        device_memory_padding=211025920, host_memory_padding=140544512,
    )
    # INTERP_TRIANGULAR is the closest antialiased match to torchvision's
    # Resize(antialias=True); pixel values are not bit-identical (verified in
    # dali_smoke_test.py: means agree to ~1e-2, which is expected of a training
    # pipeline, not a numerically-reproducible one).
    images = fn.resize(images, resize_shorter=256, interp_type=types.INTERP_TRIANGULAR)
    # Must mirror FastImageNetDataset exactly: random 224 crop, random horizontal
    # flip, and uint8 output. The scale to [0,1] is deliberately NOT applied here
    # — train_rtx5090.py's loss_fn does it on the GPU for both pipelines, so
    # normalizing here as well would divide by 255 twice.
    images = fn.crop_mirror_normalize(
        images,
        crop=(224, 224),
        crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
        crop_pos_y=fn.random.uniform(range=(0.0, 1.0)),
        mirror=fn.random.coin_flip(probability=0.5),
        mean=0.0, std=1.0,
        output_layout="HWC", dtype=types.UINT8,
    )
    return images, lbls.gpu()


def build_dali_jax_iterator(data_dir, batch_size, num_threads, checkpoint_dir, device_id=0, seed=0):
    paths, labels, class_to_idx = build_file_label_lists(data_dir)
    paths, labels = filter_corrupt_files(paths, labels, checkpoint_dir)

    pipe = imagenet_pipeline(
        files=paths, labels=labels, shuffle_seed=seed, seed=seed,
        batch_size=batch_size, num_threads=num_threads, device_id=device_id,
    )
    iterator = DALIGenericIterator(
        [pipe], output_map=["images", "labels"],
        reader_name="Reader",
        last_batch_policy=LastBatchPolicy.DROP,  # replicates drop_last=True
        auto_reset=True,
    )
    return iterator, len(paths), class_to_idx
