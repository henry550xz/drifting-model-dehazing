from pathlib import Path

import torch
from torch.utils.data import DataLoader

from TrainingDeHazing import HazyMNIST
from utils import save_image_grid


def main():
    out_dir = Path("./outputs/dehaze")
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 8
    img_size = 32

    dataset = HazyMNIST(root="./data/mnist", train=True, img_size=img_size)
    loader = DataLoader(dataset, batch_size=n, shuffle=False)
    x_clean, x_hazy, _ = next(iter(loader))

    # Interleave pairs so nrow=2 gives: left col = clean, right col = hazy.
    pairs = torch.stack([x_clean, x_hazy], dim=1).reshape(2 * n, 1, img_size, img_size)

    out_path = out_dir / "fog_demo.png"
    save_image_grid(pairs, str(out_path), nrow=2)

    print(f"Saved {out_path}  ({out_path.stat().st_size} bytes)")
    print("Layout: left column = clean  |  right column = hazy")


if __name__ == "__main__":
    main()
