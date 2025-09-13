# Medical Image Captioning with ViT-GPT2 + LoRA (ROCOv2 Dataset)

This project implements **medical image captioning** using the **ViT-GPT2** model with **LoRA fine-tuning** on the **ROCOv2 dataset**.  
The goal is to generate textual descriptions for medical images, such as X-rays and CT scans.

---

##  Project Steps

### 1. Data Loading and Preprocessing
- Load medical images and captions from **ROCOv2**.
- Resize images to **224×224 pixels**.
- Process captions using **GPT-2 tokenizer**.
- Split dataset into **training and test sets**.
- Use **10% of training data** for faster experimentation on limited GPU resources.

### 2. Model Architecture (ViT-GPT2 + LoRA)
- **Encoder**: Vision Transformer (ViT)
- **Decoder**: GPT-2 for text generation
- **LoRA Fine-Tuning**: Applied to GPT-2 attention layers (`c_attn`, `c_proj`)  
  → reduces trainable parameters and speeds up training.

### 3. Training
- Framework:  Hugging Face Transformers
- **Optimizer**: AdamW  
- **Scheduler**: Linear  
- **Batch size**: 8  
- **Epochs**: 6 (on 10% of data)  
- **Mixed precision (fp16)** training enabled  
- Save model checkpoints after each epoch.

### 4. Evaluation
- Generate captions for test images.
- **Metrics**:
  - BLEU (1–4)
  - METEOR
  - ROUGE-L
  - CIDEr
- Save results in `evaluation_results.csv`.

### 5. Results (10% Training Data)

| Metric   | Value  |
|----------|--------|
| BLEU-1   | Low    |
| BLEU-2   | Low    |
| METEOR   | Low    |
| ROUGE-L  | Low    |
| CIDEr    | Low    |

 Current results are limited due to using only **10% of training data**.  
Performance is expected to improve significantly with larger datasets and more epochs.

### 6. Future Improvements
- Train with **larger data (>10%)**.
- Tune **hyperparameters** and increase epochs.
- Apply **data augmentation** for medical images.
- Explore **alternative architectures** (e.g., BLIP, LLaVA-Med).

---

## 📊 Dataset
**ROCOv2: Radiology Objects in Context Version 2**  
- 79,789 radiological images with captions and clinical concepts  
- Seven clinical modalities, manually curated medical concepts  
- Suitable for image captioning and multi-label classification  

 **Dataset Link**: [ROCOv2 Dataset](https://zenodo.org/records/10821435)

###  Citation (APA)
> Johannes Rückert, Louise Bloch, Raphael Brüngel, Ahmad Idrissi-Yaghir, Henning Schäfer, Cynthia S. Schmidt, Sven Koitka, Obioma Pelka, Asma Ben Abacha, Alba Garcia Seco de Herrera, Henning Müller, Peter A. Horn, Felix Nensa, & Christoph M. Friedrich. (2023). ROCOv2: Radiology Objects in COntext Version 2, An Updated Multimodal Image Dataset [Data set]. *Scientific Data (2.0.1)*. Zenodo. https://doi.org/10.5281/zenodo.10821435

---

##  Dataset License
- **License**: Creative Commons Attribution Non-Commercial 4.0 International (CC BY-NC 4.0)  
- The dataset may be used for **research and educational purposes only**.  
- **Commercial use is not allowed**.  
- Users must **cite the original dataset** in publications and projects.  
- [Official Dataset Page](https://zenodo.org/records/10821435)

---

##  Requirements
```bash
torch>=2.0
transformers>=4.10.3
datasets>=2.20.0
peft
bitsandbytes
pycocoevalcap
evaluate
sacrebleu
nltk
rouge-score
numpy
pandas
matplotlib
tqdm
Pillow

