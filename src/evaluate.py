"""Evaluates a fine-tuned ViT-GPT2 model on the ROCOv2 test split.

Computes BLEU-1..4, METEOR, ROUGE-L and CIDEr, and writes per-sample
predictions to a CSV file.

Example:
    python src/evaluate.py \
        --data-dir data \
        --checkpoint vit-gpt2-roco-lora \
        --num-samples 1000 \
        --output evaluation_results.csv
"""

import argparse
import os

import pandas as pd
import torch
from PIL import Image
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from tqdm import tqdm
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ViT-GPT2 on ROCOv2")
    parser.add_argument("--data-dir", default="data", help="Root folder containing the dataset")
    parser.add_argument(
        "--test-images-subdir",
        default="datasets/test_ROCOv2/test",
        help="Subfolder (under --data-dir) with test images",
    )
    parser.add_argument("--test-csv", default="test_captions.csv", help="Filename of the test captions CSV")
    parser.add_argument("--checkpoint", required=True, help="Path to the fine-tuned model/checkpoint directory")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of test rows to evaluate")
    parser.add_argument("--max-length", type=int, default=64, help="Max generated caption length")
    parser.add_argument("--output", default="evaluation_results.csv", help="Where to save per-sample results")
    return parser.parse_args()


def generate_caption(model, processor, tokenizer, device, image_path, max_length):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        outputs = model.generate(pixel_values, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def main():
    args = parse_args()

    test_images_path = os.path.join(args.data_dir, args.test_images_subdir)
    test_csv_path = os.path.join(args.data_dir, args.test_csv)

    df_test = pd.read_csv(test_csv_path)
    df_test_sample = df_test.head(args.num_samples).reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint).to(device)
    processor = ViTImageProcessor.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model.eval()

    image_ids, gt_captions, pred_captions = [], [], []

    for _, row in tqdm(df_test_sample.iterrows(), total=len(df_test_sample)):
        image_id = row["ID"]
        if not image_id.lower().endswith(".jpg"):
            image_id += ".jpg"
        image_path = os.path.join(test_images_path, image_id)

        try:
            pred_caption = generate_caption(model, processor, tokenizer, device, image_path, args.max_length)
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping {image_id}: {exc}")
            continue

        image_ids.append(image_id)
        gt_captions.append(row["Caption"])
        pred_captions.append(pred_caption)

    res_dict = {img_id: [pred] for img_id, pred in zip(image_ids, pred_captions)}
    gts_dict = {img_id: [gt] for img_id, gt in zip(image_ids, gt_captions)}

    bleu_scores, _ = Bleu(4).compute_score(gts_dict, res_dict)
    cider_score, _ = Cider().compute_score(gts_dict, res_dict)
    meteor_score, _ = Meteor().compute_score(gts_dict, res_dict)
    rouge_score, _ = Rouge().compute_score(gts_dict, res_dict)

    print("\nEvaluation results:")
    print(f"BLEU-1 : {bleu_scores[0]:.4f}")
    print(f"BLEU-2 : {bleu_scores[1]:.4f}")
    print(f"BLEU-3 : {bleu_scores[2]:.4f}")
    print(f"BLEU-4 : {bleu_scores[3]:.4f}")
    print(f"CIDEr  : {cider_score:.4f}")
    print(f"METEOR : {meteor_score:.4f}")
    print(f"ROUGE-L: {rouge_score:.4f}")

    results_df = pd.DataFrame(
        {
            "image_id": image_ids,
            "ground_truth": gt_captions,
            "predicted": pred_captions,
        }
    )
    results_df.to_csv(args.output, index=False)
    print(f"\nPer-sample results saved to: {args.output}")


if __name__ == "__main__":
    main()
