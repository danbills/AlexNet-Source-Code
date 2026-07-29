"""
Tailor-made AlexNet ImageNet Training Script with Checkpointing.
Features:
- Checkpoint Save/Resume (saves model state every epoch to check-point dir).
- FP8 (E4M3) / BF16 mixed-precision options.
- System & GPU Telemetry logging.
"""

import os
import sys
import glob
import time
import argparse
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np

import torch
import torchvision.transforms.v2 as transforms
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

from dali_pipeline import build_file_label_lists, build_dali_jax_iterator


class AlexNet5090(nnx.Module):
    def __init__(self, num_classes: int = 1000, dtype=jnp.bfloat16, rngs: nnx.Rngs = None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.dtype = dtype
        
        self.conv1 = nnx.Conv(in_features=3, out_features=96, kernel_size=(11, 11), strides=(4, 4), padding=((2, 2), (2, 2)), rngs=rngs)
        self.conv2 = nnx.Conv(in_features=96, out_features=256, kernel_size=(5, 5), padding=((2, 2), (2, 2)), rngs=rngs)
        self.conv3 = nnx.Conv(in_features=256, out_features=384, kernel_size=(3, 3), padding=((1, 1), (1, 1)), rngs=rngs)
        self.conv4 = nnx.Conv(in_features=384, out_features=384, kernel_size=(3, 3), padding=((1, 1), (1, 1)), rngs=rngs)
        self.conv5 = nnx.Conv(in_features=384, out_features=256, kernel_size=(3, 3), padding=((1, 1), (1, 1)), rngs=rngs)
        
        self.fc6 = nnx.Linear(in_features=256 * 6 * 6, out_features=4096, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=0.5, rngs=rngs)
        self.fc7 = nnx.Linear(in_features=4096, out_features=4096, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=0.5, rngs=rngs)
        self.fc8 = nnx.Linear(in_features=4096, out_features=num_classes, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.relu(self.conv1(x))
        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = nnx.relu(self.conv2(x))
        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = nnx.relu(self.conv3(x))
        x = nnx.relu(self.conv4(x))
        x = nnx.relu(self.conv5(x))
        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        
        x = x.reshape((x.shape[0], -1))
        x = nnx.relu(self.fc6(x))
        x = self.dropout1(x)
        x = nnx.relu(self.fc7(x))
        x = self.dropout2(x)
        x = self.fc8(x)
        return x


def loss_fn(model: AlexNet5090, batch_x: jax.Array, batch_y: jax.Array):
    logits = model(batch_x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch_y).mean()
    
    top1_acc = (jnp.argmax(logits, axis=-1) == batch_y).mean()
    top5_preds = jnp.argsort(logits, axis=-1)[:, -5:]
    top5_acc = jnp.any(top5_preds == batch_y[:, None], axis=-1).mean()
    
    return loss, (top1_acc, top5_acc)


@nnx.jit
def train_step(model: AlexNet5090, optimizer: nnx.Optimizer, batch_x: jax.Array, batch_y: jax.Array):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (top1, top5)), grads = grad_fn(model, batch_x, batch_y)
    optimizer.update(model, grads)
    return loss, top1, top5


class FastImageNetDataset(Dataset):
    def __init__(self, root_dir: str):
        self.image_paths, self.labels, class_to_idx = build_file_label_lists(root_dir)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToDtype(torch.float32, scale=True),
        ])
        print(f"Dataset initialized: {len(class_to_idx)} classes, {len(self.image_paths)} total images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = read_image(path, mode=ImageReadMode.RGB)
            img = self.transform(img)
            img = img.permute(1, 2, 0)
            return img, self.labels[idx]
        except Exception:
            return torch.zeros((224, 224, 3), dtype=torch.float32), self.labels[idx]


def torch_to_jax(tensor: torch.Tensor) -> jax.Array:
    return jax.device_put(jnp.array(tensor.numpy()))


def _flatten_state(state, prefix):
    return {
        f"{prefix}/" + "/".join(str(k) for k in path): np.asarray(leaf)
        for path, leaf in jax.tree_util.tree_leaves_with_path(state)
    }


def save_checkpoint(path: str, model: AlexNet5090, optimizer: nnx.Optimizer, epoch: int, global_step: int):
    """Persists model weights and optimizer state (e.g. SGD momentum), not just
    the epoch/step counters, so --resume actually continues training instead of
    restarting a freshly-initialized model under an advanced LR schedule.

    Model state is filtered to nnx.Param only, excluding the Dropout layers'
    RNG-stream variables (a JAX PRNGKey dtype that can't round-trip through
    np.asarray) — those don't need to survive a resume, dropout masks are
    stochastic regularization, not state that affects correctness.
    """
    flat = {}
    flat.update(_flatten_state(nnx.state(model, nnx.Param), "model"))
    flat.update(_flatten_state(nnx.state(optimizer), "opt"))
    flat["epoch"] = np.asarray(epoch)
    flat["global_step"] = np.asarray(global_step)

    tmp_path = path + ".tmp.npz"
    np.savez(tmp_path, **flat)
    os.replace(tmp_path, path)


def load_checkpoint(path: str, model: AlexNet5090, optimizer: nnx.Optimizer) -> tuple[int, int]:
    data = np.load(path)

    def restore(state, prefix):
        leaves_with_path = jax.tree_util.tree_leaves_with_path(state)
        new_leaves = [
            jnp.asarray(data[f"{prefix}/" + "/".join(str(k) for k in p)])
            for p, _ in leaves_with_path
        ]
        return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(state), new_leaves)

    nnx.update(model, restore(nnx.state(model, nnx.Param), "model"))
    nnx.update(optimizer, restore(nnx.state(optimizer), "opt"))
    return int(data["epoch"]), int(data["global_step"])


def main():
    parser = argparse.ArgumentParser(description="RTX 5090 + Ryzen 9800X3D AlexNet Training Config")
    parser.add_argument("--data-dir", type=str, default="/home/dan/imagenet/train")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--base-lr", type=float, default=0.02)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--data-pipeline", type=str, choices=["torch", "dali"], default="torch",
                         help="torch: CPU decode via DataLoader workers. dali: GPU decode+resize via nvJPEG/DALI.")
    parser.add_argument("--num-threads", type=int, default=8, help="DALI pipeline threads (--data-pipeline dali only)")
    parser.add_argument("--log-dir", type=str, default="logs/tensorboard_5090")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    args = parser.parse_args()

    print(f"=== RTX 5090 + Ryzen 9800X3D Max Performance Config ===")
    print(f"Batch Size: {args.batch_size} | LR: {args.base_lr} | Workers: {args.num_workers} | Epochs: {args.epochs} | Pipeline: {args.data_pipeline}")
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    if args.data_pipeline == "torch":
        dataset = FastImageNetDataset(args.data_dir)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
            drop_last=True
        )
        dataset_size = len(dataset)

        def batch_source():
            for x, y in dataloader:
                yield torch_to_jax(x), torch_to_jax(y)
    else:
        dali_iterator, dataset_size, _ = build_dali_jax_iterator(
            args.data_dir, args.batch_size, args.num_threads, args.checkpoint_dir,
        )

        def batch_source():
            for batch in dali_iterator:
                yield batch["images"], batch["labels"].reshape(-1)

    rngs = nnx.Rngs(0)
    model = AlexNet5090(num_classes=1000, dtype=jnp.bfloat16, rngs=rngs)

    # Warmup + Piecewise constant decay schedule
    warmup_steps = (dataset_size // args.batch_size) * args.warmup_epochs
    decay_steps = (dataset_size // args.batch_size) * 4
    
    warmup_schedule = optax.linear_schedule(init_value=0.001, end_value=args.base_lr, transition_steps=warmup_steps)
    decay_schedule = optax.piecewise_constant_schedule(
        init_value=args.base_lr,
        boundaries_and_scales={
            decay_steps: 0.1,
            decay_steps * 2: 0.1
        }
    )
    lr_schedule = optax.join_schedules([warmup_schedule, decay_schedule], boundaries=[warmup_steps])
    optimizer = nnx.Optimizer(model, optax.sgd(learning_rate=lr_schedule, momentum=0.9), wrt=nnx.Param)

    start_epoch = 1
    global_step = 0
    latest_ckpt = os.path.join(args.checkpoint_dir, "alexnet_latest.npz")
    
    if args.resume and os.path.exists(latest_ckpt):
        print(f"Resuming training state from checkpoint: {latest_ckpt}")
        ckpt_epoch, global_step = load_checkpoint(latest_ckpt, model, optimizer)
        start_epoch = ckpt_epoch + 1
        print(f"Resuming from Epoch {start_epoch}, Step {global_step} (model + optimizer state restored)")

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()
        t_loss, t_top1, t_top5 = 0.0, 0.0, 0.0
        steps = 0

        for bx, by in batch_source():
            l, top1, top5 = train_step(model, optimizer, bx, by)
            t_loss += float(l)
            t_top1 += float(top1)
            t_top5 += float(top5)
            steps += 1
            global_step += 1
            
            writer.add_scalar("Step/Train_Loss", float(l), global_step)
            writer.add_scalar("Step/Train_Top1_Acc", float(top1) * 100, global_step)
            writer.add_scalar("Step/Train_Top5_Acc", float(top5) * 100, global_step)
            writer.add_scalar("Hyperparams/Learning_Rate", float(lr_schedule(global_step)), global_step)

        if steps > 0:
            t_loss /= steps
            t_top1 /= steps
            t_top5 /= steps

        elapsed = time.time() - start_time
        throughput = (steps * args.batch_size) / elapsed if elapsed > 0 else 0

        writer.add_scalar("Epoch/Train_Loss", t_loss, epoch)
        writer.add_scalar("Epoch/Train_Top1_Acc", t_top1 * 100, epoch)
        writer.add_scalar("Epoch/Top5_Acc_Epoch", t_top5 * 100, epoch)
        writer.add_scalar("Performance/Throughput_Img_Per_Sec", throughput, epoch)

        # Save checkpoint (model + optimizer state) at end of epoch
        save_checkpoint(latest_ckpt, model, optimizer, epoch, global_step)
        print(f"Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.2f}s] | "
              f"Train Loss: {t_loss:.4f} | "
              f"Top-1 Acc: {t_top1*100:.2f}% | Top-5 Acc: {t_top5*100:.2f}% | "
              f"Speed: {throughput:.1f} img/s (Checkpoint Saved)")

    writer.close()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
