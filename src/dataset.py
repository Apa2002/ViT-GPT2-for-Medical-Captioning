"""PyTorch Dataset for the ROCOv2 medical image-captioning dataset."""

import os

from PIL import Image
from torch.utils.data import Dataset


class ROCOv2Dataset(Dataset):
    """Loads ROCOv2 images and their captions, ready for ViT-GPT2 training.

    Args:
        dataframe: DataFrame with at least the columns ``ID`` and ``Caption``.
        image_dir: Directory containing the ``.jpg`` images referenced by ``ID``.
        processor: A ``ViTImageProcessor`` (or compatible) used to preprocess images.
        tokenizer: A GPT-2 compatible tokenizer used to encode captions.
        max_length: Maximum token length for the tokenized caption.
    """

    def __init__(self, dataframe, image_dir, processor, tokenizer, max_length=64):
        self.data = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row["ID"]

        if not image_id.lower().endswith(".jpg"):
            image_id += ".jpg"

        image_path = os.path.join(self.image_dir, image_id)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()

        caption = row["Caption"]
        input_ids = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).input_ids.squeeze()

        return {
            "pixel_values": pixel_values,
            "labels": input_ids,
        }
