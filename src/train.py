"""Fine-tunes ViT-GPT2 (LoRA) on the ROCOv2 medical captioning dataset.

Example:
    python src/train.py \
        --data-dir data \
        --output-dir vit-gpt2-roco-lora \
        --train-frac 0.1 \
        --epochs 6
"""

import argparse
import os

import pandas as pd
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from dataset import ROCOv2Dataset
from model import DEFAULT_MODEL_NAME, build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT-GPT2 + LoRA on ROCOv2")
    parser.add_argument("--data-dir", default="data", help="Root folder containing the dataset")
    parser.add_argument(
        "--train-images-subdir",
        default="datasets/train_ROCOv2/train",
        help="Subfolder (under --data-dir) with training images",
    )
    parser.add_argument("--train-csv", default="train_captions.csv", help="Filename of the training captions CSV")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Base HF model to fine-tune")
    parser.add_argument("--output-dir", default="vit-gpt2-roco-lora", help="Where to save checkpoints")
    parser.add_argument("--train-frac", type=float, default=0.1, help="Fraction of training data to sample")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    train_images_path = os.path.join(args.data_dir, args.train_images_subdir)
    train_csv_path = os.path.join(args.data_dir, args.train_csv)

    df_train = pd.read_csv(train_csv_path)
    df_train = df_train.sample(frac=args.train_frac, random_state=args.seed).reset_index(drop=True)

    model, processor, tokenizer = build_model(model_name=args.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = ROCOv2Dataset(df_train, train_images_path, processor, tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        logging_dir="./logs",
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"\nTraining complete. Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
