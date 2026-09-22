"""Displays a handful of test images with ground-truth vs. predicted captions.

Example:
    python src/visualize.py \
        --results evaluation_results.csv \
        --images-dir data/datasets/test_ROCOv2/test \
        --num-examples 8
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize ROCOv2 caption predictions")
    parser.add_argument("--results", required=True, help="CSV produced by evaluate.py")
    parser.add_argument("--images-dir", required=True, help="Folder containing the test images")
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", default=None, help="If set, saves the figure instead of showing it")
    return parser.parse_args()


def main():
    args = parse_args()
    results_df = pd.read_csv(args.results)
    sample_df = results_df.sample(n=args.num_examples, random_state=args.seed).reset_index(drop=True)

    plt.figure(figsize=(16, args.num_examples * 2.5))

    for i in range(args.num_examples):
        image_path = os.path.join(args.images_dir, sample_df.loc[i, "image_id"])
        image = Image.open(image_path).convert("RGB")

        plt.subplot(args.num_examples, 1, i + 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title(
            f"Ground Truth: {sample_df.loc[i, 'ground_truth']}\nPredicted: {sample_df.loc[i, 'predicted']}",
            fontsize=10,
            loc="left",
        )

    plt.tight_layout()

    if args.save_path:
        plt.savefig(args.save_path, bbox_inches="tight")
        print(f"Figure saved to: {args.save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
