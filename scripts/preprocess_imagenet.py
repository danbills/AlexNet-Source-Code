"""One-time offline pre-resize of the ImageNet training set.

Writes a parallel directory tree of JPEGs whose *shorter side is exactly 256*,
which is what train_rtx5090.py's RandomCrop(224) + RandomHorizontalFlip expects.

Why this exists: decoding full-resolution ImageNet JPEGs every epoch is the
training loop's dominant cost. Measured on 400 real images from n01440764:

    original full-res   110.5 KB/img    920 img/s/core decode
    shorter-side 256     27.0 KB/img   2857 img/s/core decode   (4.1x / 3.1x)

The size reduction matters as much as the decode speedup: it takes the 144-class
subset from 22 GB to ~5.4 GB, which fits entirely in page cache on a 28 GB
machine, so disk I/O disappears after the first epoch. At the full 1000 classes
it is the difference between ~140 GB and ~35 GB read per epoch.

The speed of the resize itself comes from Image.draft(), which asks libjpeg for a
DCT-domain scaled decode (1/2, 1/4, 1/8) and so skips most of the IDCT work for
pixels we are about to throw away.

Usage:
    python scripts/preprocess_imagenet.py \
        --in-dir /home/dan/imagenet/train --out-dir /home/dan/imagenet/train256

Resumable: files already present in --out-dir are skipped, so it is safe to
re-run after interrupting it or after extracting more class tars.
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dali_pipeline import build_file_label_lists  # noqa: E402  (shared class->idx ordering)


def _resize_one(job):
    """Returns (ok, skipped, in_bytes, out_bytes). Never raises: a handful of
    ImageNet JPEGs are truncated, and one bad file must not kill the pool."""
    src, dst, size, quality = job
    try:
        if os.path.exists(dst):
            return True, True, 0, 0

        with Image.open(src) as im:
            # DCT-domain scaled decode: the whole reason this script is fast.
            # draft() only ever picks a scale that stays >= the requested size.
            im.draft("RGB", (size, size))
            im = im.convert("RGB")

            w, h = im.size
            scale = size / min(w, h)
            # Note the missing `if scale < 1` guard is deliberate: images whose
            # shorter side is under 256 must be scaled UP, or RandomCrop(224)
            # fails on them later. ImageNet does contain such images.
            new_size = (max(size, round(w * scale)), max(size, round(h * scale)))
            if new_size != (w, h):
                im = im.resize(new_size, Image.BILINEAR)

            tmp = dst + ".tmp"
            im.save(tmp, "JPEG", quality=quality)
            os.replace(tmp, dst)

        return True, False, os.path.getsize(src), os.path.getsize(dst)
    except Exception as e:
        print(f"  skipping {src}: {type(e).__name__}: {e}", file=sys.stderr)
        return False, False, 0, 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", default="/home/dan/imagenet/train")
    p.add_argument("--out-dir", default="/home/dan/imagenet/train256")
    p.add_argument("--size", type=int, default=256, help="target shorter side")
    p.add_argument("--quality", type=int, default=90, help="output JPEG quality")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args()

    paths, _labels, class_to_idx = build_file_label_lists(args.in_dir)
    if not paths:
        sys.exit(f"No images found under {args.in_dir} (expected n*/ class dirs of *.JPEG)")
    print(f"Found {len(paths)} images in {len(class_to_idx)} classes under {args.in_dir}")
    print(f"Writing shorter-side-{args.size} q{args.quality} JPEGs to {args.out_dir} "
          f"using {args.workers} workers")

    for cls in class_to_idx:
        os.makedirs(os.path.join(args.out_dir, cls), exist_ok=True)

    jobs = []
    for src in paths:
        rel = os.path.relpath(src, args.in_dir)
        jobs.append((src, os.path.join(args.out_dir, rel), args.size, args.quality))

    t0 = time.time()
    n_ok = n_skip = n_bad = 0
    in_bytes = out_bytes = 0
    with mp.Pool(args.workers) as pool:
        for i, (ok, skipped, ib, ob) in enumerate(pool.imap_unordered(_resize_one, jobs, chunksize=64), 1):
            n_ok += ok
            n_skip += skipped
            n_bad += not ok
            in_bytes += ib
            out_bytes += ob
            if i % 20000 == 0:
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(jobs)}  {rate:.0f} img/s  eta {(len(jobs) - i) / rate:.0f}s")

    dt = time.time() - t0
    n_new = n_ok - n_skip
    print(f"\nDone in {dt:.1f}s ({len(jobs) / dt:.0f} img/s): "
          f"{n_new} written, {n_skip} already present, {n_bad} unreadable")
    if n_new and in_bytes:
        print(f"Size: {in_bytes / n_new / 1024:.1f} KB/img -> {out_bytes / n_new / 1024:.1f} KB/img "
              f"({in_bytes / out_bytes:.1f}x smaller)")
    print(f"\nNow train with:  --data-dir {args.out_dir}")


if __name__ == "__main__":
    main()
