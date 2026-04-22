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
from train import TrainConfig

@dataclass
class InferenceConfig:
    seed: int = 42
    checkpoint_path: str = "./mnist_flow_matching/step_010000.pt"
    work_dir: str = "./inference"
    num_samples: int = 64
    dataset_dir: str = "./dataset"
    num_steps: int = 100
    t_start: float = 1.0
    t_end: float = 1e-2
    accelerator: str = "auto"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
                "TPU accelerator requested but torch_xla is not installed on this node."
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

def denormalize_mnist(x: torch.Tensor) -> torch.Tensor:
    """Convert normalized MNIST tensors back to [0, 1] for visualization."""
    x = x * 0.3081 + 0.1307
    return x.clamp(0.0, 1.0)

def generate_samples(model_net, device, num_samples, num_steps, t_start, t_end, use_xla=False):
    model_net.eval()
    xm = None
    if use_xla:
        import torch_xla.core.xla_model as xm  # type: ignore
    with torch.no_grad():
        z = torch.randn(num_samples, 1, 28, 28, device=device)
        t_grid = torch.linspace(t_start, t_end, steps=num_steps, device=device)
        for i in range(num_steps - 1):
            dt = t_grid[i + 1] - t_grid[i] # move from t_grid[i] to t_grid[i+1]

            t_in = torch.full((num_samples,), t_grid[i].item(), device=device)
            pred_v = model_net(z, t_in)

            z = z + dt * pred_v
            if use_xla:
                xm.mark_step()
        #t_grid = -torch.cos(t_grid*100)*0.5 + 0.5 # try this out
        #for i in range(num_steps - 1):
            #dt = t_grid[i + 1] - t_grid[i] # move from t_grid[i] to t_grid[i+1]

            #t_in = torch.full((num_samples,), t_grid[i].item(), device=device)
            #pred_v = model_net(z, t_in)

            #z = z + dt * pred_v
    return z

def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained DiT flow-matching model.")
    parser.add_argument("--checkpoint-path", type=str, default="./mnist_flow_matching/step_010000.pt")
    parser.add_argument("--work-dir", type=str, default="./inference")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--t-start", type=float, default=1.0)
    parser.add_argument("--t-end", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expert-name", type=str, default="expert")
    parser.add_argument("--accelerator", type=str, default="auto", choices=["auto", "gpu", "tpu", "cpu"])
    args = parser.parse_args()

    config = InferenceConfig(
        seed=args.seed,
        checkpoint_path=args.checkpoint_path,
        work_dir=args.work_dir,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        t_start=args.t_start,
        t_end=args.t_end,
        accelerator=args.accelerator,
    )
    set_seed(config.seed)
    
    device, use_xla = resolve_device(config.accelerator)
    
    # Load model
    model_net = model.DiT().to(device)
    checkpoint = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
    model_net.load_state_dict(checkpoint['model_state_dict'])
    
    # Generate samples
    samples = generate_samples(
        model_net,
        device,
        num_samples=config.num_samples,
        num_steps=config.num_steps,
        t_start=config.t_start,
        t_end=config.t_end,
        use_xla=use_xla,
    )
    samples = samples.view(-1, 1, 28, 28)
    
    # Unpatchify and denormalize for visualization
    samples = denormalize_mnist(samples.cpu())
    
    # Save generated samples as images
    os.makedirs(config.work_dir, exist_ok=True)
    for i in range(samples.size(0)):
        plt.imsave(os.path.join(config.work_dir, f"sample_{i:03d}.png"), samples[i].squeeze(), cmap='gray')

    print(
        f"[{args.expert_name}] Generated {config.num_samples} samples "
        f"saved to {config.work_dir}"
    )

if __name__ == "__main__":
    main()
