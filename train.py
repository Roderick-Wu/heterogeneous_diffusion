import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataclasses import dataclass
from pathlib import Path
import model
import data
import matplotlib.pyplot as plt
import time
import logging
import random

@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 1e-4
    steps: int = 10000
    checkpoint_freq: int = 100
    log_freq: int = 10
    dataset_dir: str = "./dataset"
    work_dir: str = "./mnist_flow_matching"
    resume_from: str = None
    #resume_from: str = "./mnist_flow_matching/step_010000.pt"
    val_split: float = 0.1 
    num_workers: int = 0
    shard_index: int = 0
    num_shards: int = 1
    expert_name: str = "expert"
    accelerator: str = "auto"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(accelerator: str):
    accelerator = accelerator.lower()

    if accelerator == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU accelerator requested but CUDA is not available.")
        return torch.device("cuda"), False

    if accelerator == "cpu":
        return torch.device("cpu"), False

    if accelerator == "tpu":
        try:
            import torch_xla.core.xla_model as xm  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "TPU accelerator requested but torch_xla import failed on this node. "
                "This is often a torch/torch_xla ABI mismatch. "
                f"Import error: {exc!r}"
            ) from exc
        return xm.xla_device(), True

    if accelerator == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), False
        try:
            import torch_xla.core.xla_model as xm  # type: ignore
            return xm.xla_device(), True
        except ImportError:
            return torch.device("cpu"), False

    raise ValueError("accelerator must be one of: auto, gpu, tpu, cpu")

def setup_logger(work_dir: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(work_dir, "training.log")),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_checkpoint(step, model, optimizer, rng_states, loss_data, loader_state, config, checkpoint_dir):
    """Saves full state to allow exact resumption."""
    path = os.path.join(checkpoint_dir, f"step_{step:06d}.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rng_states': rng_states,
        'loss_data': loss_data, # for future plotting if it dies
        'loader_state': loader_state,
        'config': config
    }, path)
    logging.info(f"Saved checkpoint to {path}")

def load_checkpoint(path, device, model, optimizer):
    checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    rng_states = checkpoint['rng_states']
    random.setstate(rng_states['python'])
    np.random.set_state(rng_states['numpy'])
    torch.set_rng_state(rng_states['torch'])
    if device.type == 'cuda':
        if rng_states.get('cuda_all') is not None:
            torch.cuda.set_rng_state_all(rng_states['cuda_all'])
        elif rng_states.get('cuda') is not None:
            torch.cuda.set_rng_state(rng_states['cuda'])
        
    return (
        checkpoint['step'],
        checkpoint['loss_data']['train'],
        checkpoint['loss_data']['val'],
        checkpoint.get('loader_state', None),
    )

def train(config, model, device, loss_plot_path, use_xla=False):
    model = model.to(device)
    xm = None
    if use_xla:
        import torch_xla.core.xla_model as xm  # type: ignore

    if config.num_workers != 0:
        logger.warning(
            "num_workers != 0 can break exact checkpoint reproducibility due to DataLoader prefetch. "
            "Use num_workers=0 for exact loss matching after resume."
        )
    
    dataset = data.MNISTDataset(save_path=config.dataset_dir, train=True)
    dataset = data.take_dataset_shard(
        dataset,
        shard_index=config.shard_index,
        num_shards=config.num_shards,
    )
    train_dataset, val_dataset = data.shuffle_and_split_dataset(dataset, val_split=config.val_split)

    #dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    #train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    #val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    train_batch_sampler = data.StatefulBatchSampler(
        dataset_size=len(train_dataset),
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=True,
    )
    val_batch_sampler = data.StatefulBatchSampler(
        dataset_size=len(val_dataset),
        batch_size=config.batch_size,
        seed=config.seed + 1,
        shuffle=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=config.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_sampler=val_batch_sampler,
        num_workers=config.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    step = 0
    train_losses, val_losses = [], []

    # import pdb; pdb.set_trace()
    if config.resume_from is not None:
        step, train_losses, val_losses, loader_state = load_checkpoint(
            config.resume_from, 
            device, 
            model, 
            optimizer
        )
        if loader_state is not None:
            if 'train_sampler' in loader_state:
                train_batch_sampler.load_state_dict(loader_state['train_sampler'])
            if 'val_sampler' in loader_state:
                val_batch_sampler.load_state_dict(loader_state['val_sampler'])
        step += 1 ##
        logger.info(f"Resumed from checkpoint: {config.resume_from} at step {step}")

    os.makedirs(config.work_dir, exist_ok=True)
    logger.info(f"Checkpoint directory: {config.work_dir}")
    logger.info(
        f"Expert {config.expert_name}: shard {config.shard_index + 1}/{config.num_shards}, "
        f"train_size={len(train_dataset)}, val_size={len(val_dataset)}"
    )

    train_iter = iter(train_dataloader)
    val_iter = iter(val_dataloader)

    while step < config.steps:

        # Train
        model.train()

        try:
            (x, y) = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            print(f"Starting new training epoch at step {step}")
            (x, y) = next(train_iter)

        x = x.to(device)

        eps = torch.randn_like(x, device=device)
        t = torch.rand(x.size(0), device=device)

        xt = x * (1 - t.view(-1, 1, 1, 1)) + eps * t.view(-1, 1, 1, 1) # add noise to original

        target_v = eps - x

        pred_v = model(xt, t) # model predicts the flow

        loss = F.mse_loss(pred_v, target_v)
        optimizer.zero_grad()
        loss.backward()
        if use_xla:
            xm.optimizer_step(optimizer, barrier=False)
            xm.mark_step()
        else:
            optimizer.step()

        train_losses.append(loss.item())
        if step % config.log_freq == 0:
            logger.info(f"Step {step}: Train Loss = {np.mean(train_losses[-config.log_freq:]):.4f}")


        # Validation
        model.eval()
        with torch.no_grad():
            try:
                (x_val, y_val) = next(val_iter)
            except StopIteration:
                val_iter = iter(val_dataloader)
                print(f"Starting new validation epoch at step {step}")
                (x_val, y_val) = next(val_iter)
                
            x_val = x_val.to(device)

            eps = torch.randn_like(x_val, device=device)
            t = torch.rand(x_val.size(0), device=device)

            xt = x_val * (1 - t.view(-1, 1, 1, 1)) + eps * t.view(-1, 1, 1, 1)

            target_v = eps - x_val

            pred_v = model(xt, t)

            loss = F.mse_loss(pred_v, target_v)
                
            val_losses.append(loss.item())

        if step % config.log_freq == 0:   
            logger.info(f"Step {step}: Val Loss = {np.mean(val_losses[-config.log_freq:]):.4f}")

        if step % config.checkpoint_freq == 0:
            loss_data = {"train": train_losses, "val": val_losses}
            save_checkpoint(
                step=step, 
                model=model, 
                optimizer=optimizer,
                rng_states={
                    'python': random.getstate(),
                    'numpy': np.random.get_state(),
                    'torch': torch.get_rng_state().detach().cpu(),
                    'cuda': torch.cuda.get_rng_state().detach().cpu() if device.type == 'cuda' else None,
                    'cuda_all': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                }, 
                loss_data=loss_data, 
                loader_state={
                    'train_sampler': train_batch_sampler.state_dict(),
                    'val_sampler': val_batch_sampler.state_dict(),
                },
                config=config, 
                checkpoint_dir=config.work_dir
            )

        step += 1

    
    loss_data = {"train": train_losses, "val": val_losses}
    save_checkpoint(
        step=step, 
        model=model, 
        optimizer=optimizer,
        rng_states={
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state().detach().cpu(),
            'cuda': torch.cuda.get_rng_state().detach().cpu() if device.type == 'cuda' else None,
            'cuda_all': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
        }, 
        loss_data=loss_data, 
        loader_state={
            'train_sampler': train_batch_sampler.state_dict(),
            'val_sampler': val_batch_sampler.state_dict(),
        },
        config=config, 
        checkpoint_dir=config.work_dir
    )

    # Plot loss curves
    plt.figure(figsize=(12, 5))
    
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.work_dir, loss_plot_path))
    logger.info(f"Loss curve saved to {os.path.join(config.work_dir, loss_plot_path)}")
    
    logger.info(f"\nTraining completed!")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DiT flow-matching model on MNIST.")
    parser.add_argument("--work-dir", type=str, default="./mnist_flow_matching")
    parser.add_argument("--dataset-dir", type=str, default="./dataset")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--checkpoint-freq", type=int, default=100)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--expert-name", type=str, default="expert")
    parser.add_argument("--accelerator", type=str, default="auto", choices=["auto", "gpu", "tpu", "cpu"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    logger = setup_logger(args.work_dir)
    device, use_xla = resolve_device(args.accelerator)
    logger.info(f"Using device: {device} (accelerator={args.accelerator}, xla={use_xla})")
    
    config = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        steps=args.steps,
        checkpoint_freq=args.checkpoint_freq,
        log_freq=args.log_freq,
        dataset_dir=args.dataset_dir,
        work_dir=args.work_dir,
        resume_from=args.resume_from,
        val_split=args.val_split,
        num_workers=args.num_workers,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        expert_name=args.expert_name,
        accelerator=args.accelerator,
    )
    set_seed(config.seed)

    # Create model
    model = model.DiT(
        in_channels=1,
        patch_size=4,
        img_size=28,
        hidden_dim=256,
        num_layers=4,
        num_heads=8
    )    

    logger.info(f"Created model")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    train(config, model, device, loss_plot_path="loss_curve.png", use_xla=use_xla)