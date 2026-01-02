import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_losses(loss_file: str, output_path: str):
   
    losses = np.load(loss_file)
    rounds = range(1, len(losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, losses, marker="o")
    plt.title("Aggregated Training Loss")
    plt.xlabel("Rounds")
    plt.ylabel("Train Loss")
    plt.grid(True)
    plt.xticks(rounds)
    plt.tight_layout()

    plot_save_path = os.path.join(output_path, "training_loss.png")
    plt.savefig(plot_save_path)
    print(f"Plot saved to {plot_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot aggregated training losses from a .npy file."
    )
    parser.add_argument(
        "--loss-file",
        type=str,
        required=True,
        help="Path to the .npy file containing the aggregated losses.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=".",
        help="Directory to save the plot. Defaults to current directory.",
    )
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    plot_losses(args.loss_file, args.output_path)
