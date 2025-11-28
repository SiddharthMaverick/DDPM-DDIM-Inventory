import os
import random
import numpy as np
import torch

# -------------------------------------------------------------
# Utility functions for seeding, directory management, and logging
# -------------------------------------------------------------

def seed_everything(seed: int = 42):
    """
    Set random seeds for reproducibility across numpy, random, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] All seeds set to {seed}")


def ensure_dir(path: str):
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)
    return path


def count_parameters(model):
    """
    Returns trainable parameter count of a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_tensor_images(tensor, filename):
    """
    Saves tensor images to disk using torchvision utilities.
    Useful for visual comparison of generated samples.
    """
    from torchvision.utils import save_image
    save_image(tensor, filename, nrow=8, normalize=True)
    print(f"[Images Saved] {filename}")
