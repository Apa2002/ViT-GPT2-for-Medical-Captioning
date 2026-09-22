"""Builds the ViT-GPT2 VisionEncoderDecoder model with a LoRA-adapted decoder."""

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor

DEFAULT_MODEL_NAME = "ydshieh/vit-gpt2-coco-en"


def build_model(
    model_name: str = DEFAULT_MODEL_NAME,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules=("c_attn", "c_proj"),
):
    """Loads the base ViT-GPT2 model and wraps its GPT-2 decoder with LoRA.

    Returns:
        (model, processor, tokenizer)
    """
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model.decoder = get_peft_model(model.decoder, peft_config)
    model.decoder.print_trainable_parameters()

    return model, processor, tokenizer
