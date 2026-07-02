# ImageCLEFmedical Caption 2026 — Synthetic Caption Prediction

This repository contains a medical image captioning pipeline for the **ImageCLEFmedical Caption 2026 — Caption Prediction Synthetical** task.

The project focuses on generating descriptive captions for synthetic medical images by combining:

- a **Vision Transformer (ViT)** image encoder,
- a **DistilGPT-2** language decoder,
- a **CUI-aware auxiliary concept prediction head**, and
- checkpoint-safe inference for producing a full submission file.

> This repository is prepared as concise supporting material for a B-ranked working-note/paper submission.  
> Dataset archives and trained checkpoints are not included because of size and access restrictions.

---

## Task Summary

The task is to generate a caption for each medical image in the synthetic validation/test split.

For each input image, the system predicts a free-text caption that should be:

- medically relevant,
- visually grounded,
- semantically close to the reference caption,
- and factual with respect to medical concepts.

The ImageCLEFmedical Caption 2026 task evaluates caption prediction using a mixture of relevance and factuality metrics, including BERTScore, ROUGE, image-caption similarity, BLEURT, MedCAT/UMLS concept matching, and AlignScore.

---

## Repository Structure

```text
.
├── README.md
├── _CLEF__26__UIT_HighDefinition__Task__Caption__Subtask__Caption_Prediction__bản_Standard__ (1).pdf
├── imageclef.ipynb
├── submission_epoch6_FINAL_(19239_cau).csv
├── Nhom 1 (clef 2025).rar
├── Nhom 2 (clef 2025).rar
├── Nhom 3 (clef 2025).rar
├── Nhóm 1 (clef 2024).rar
└── Nhóm 2 (clef 2024).rar
```

### Main Files

| File | Description |
|---|---|
| `imageclef.ipynb` | Main notebook for preprocessing, training, inference, evaluation notes, and submission generation. |
| `submission_epoch6_FINAL_(19239_cau).csv` | Final generated caption file containing 19,239 predicted captions. |
| `_CLEF__26__...Caption_Prediction...pdf` | Task/report-related PDF for the 2026 caption prediction work. |
| `README.md` | Project documentation. |

The 2024 and 2025 `.rar` files are older reference archives and are not required to understand or run the 2026 implementation.

---

## Method Overview

The final approach is a multi-task vision-language model.  
It does not only learn to generate captions; it also learns to predict medical concepts represented as CUIs.

This design was used to make captions more medically grounded instead of optimizing only for surface-level text similarity.

### Architecture

```text
Medical image
   │
   ▼
ViT-B/16 image encoder
   │
   ├── CLS image embedding ──► CUI concept head
   │
   └── Linear bridge ────────► DistilGPT-2 decoder
                                  │
                                  ▼
                           Generated caption
```

### Components

| Component | Implementation |
|---|---|
| Image encoder | `google/vit-base-patch16-224-in21k` |
| Text decoder | `distilgpt2` |
| Tokenizer | DistilGPT-2 tokenizer |
| Concept head | MLP classifier over real CUI vocabulary |
| Caption loss | GPT-2 language modeling loss |
| Concept loss | `BCEWithLogitsLoss` |
| Optimizer | `AdamW`, learning rate `5e-5` |
| Image size | `224 × 224` |
| Decoding | Beam search with `num_beams=5` |

---

## Dataset Preparation

The notebook expects the challenge data to be available from Google Drive or a local/Colab path.

Expected files include:

```text
captions.csv
images.zip
cui_to_name_synth_2026.csv
manuel_concepts/
├── train_concepts_manual.csv
└── valid_concepts_manual.csv
```

The notebook performs the following preprocessing steps:

1. Mount Google Drive.
2. Extract `images.zip`.
3. Split images into `train_images/` and `valid_images/`.
4. Read `captions.csv`.
5. Keep synthetic samples using IDs that contain `synth`.
6. Load official CUI vocabulary and manual concept annotations.
7. Build multi-hot CUI labels for each image.

Expected working structure after extraction:

```text
/content/dataset/
├── train_images/
└── valid_images/
```

---

## Training Pipeline

The final training setup uses a CUI-aware multi-task objective:

```python
loss = text_loss + CONCEPT_WEIGHT * concept_loss
```

where:

- `text_loss` trains the DistilGPT-2 decoder to generate captions,
- `concept_loss` trains the auxiliary CUI classifier,
- `CONCEPT_WEIGHT` controls the balance between language fluency and medical factuality.

The notebook experiments with different concept weights.  
A higher concept weight improves medical grounding but can reduce fluency and ROUGE-style overlap.  
The later training stage uses:

```python
CONCEPT_WEIGHT = 1.0
EPOCHS = 6
```

This setting keeps concept supervision while avoiding over-penalizing the caption decoder.

---

## Inference

For inference, the model:

1. Loads the trained epoch checkpoint.
2. Encodes each validation image with ViT.
3. Projects the image embedding into the GPT-2 embedding space.
4. Uses the projected image embedding as the prefix for caption generation.
5. Generates captions using deterministic beam search.
6. Writes predictions to a CSV submission file.

Final decoding configuration:

```python
max_new_tokens = 60
num_beams = 5
no_repeat_ngram_size = 2
early_stopping = True
do_sample = False
```

The notebook intentionally avoids temperature sampling in the final run because earlier experiments showed that sampling reduced medical factuality metrics such as MedCAT and AlignScore.

---

## Checkpoint-Safe Submission Generation

The repository includes logic for long-running inference on Colab.

The notebook can:

- scan existing partial submission files,
- collect already processed image IDs,
- resume only the remaining images,
- save progress every 3,000 captions,
- and export a final complete CSV file.

Final output:

```text
submission_epoch6_FINAL_(19239_cau).csv
```

The final submission file uses the required format:

```csv
ID,Caption
ImageCLEFmedical_Caption_2026_synth_valid_000000,"..."
ImageCLEFmedical_Caption_2026_synth_valid_000001,"..."
```

---

## Recorded Results

The notebook records the following evaluation scores for one of the main experimental runs:

| Metric | Score |
|---|---:|
| Overall | 0.3764 |
| Relevance score | 0.4452 |
| Factuality score | 0.3076 |
| BERTScore | 0.5196 |
| ROUGE | 0.2876 |
| Image-caption similarity | 0.6569 |
| BLEURT | 0.3169 |
| MedCAT | 0.2206 |
| AlignScore | 0.3947 |

A later decoding experiment with more sampling produced slightly lower factuality:

| Metric | Score |
|---|---:|
| Overall | 0.3716 |
| Relevance score | 0.4455 |
| Factuality score | 0.2977 |
| BERTScore | 0.5188 |
| ROUGE | 0.2876 |
| Image-caption similarity | 0.6586 |
| BLEURT | 0.3168 |
| MedCAT | 0.2180 |
| AlignScore | 0.3774 |

The final takeaway is that **deterministic beam search is safer for medical captioning** than more creative decoding, because medical captioning rewards factual consistency more than linguistic variety.

---

## How to Run

### 1. Install dependencies

```bash
pip install torch torchvision transformers pandas numpy pillow tqdm evaluate rouge_score bert_score
```

### 2. Prepare data

Place the dataset and metadata files in Google Drive or update the notebook paths manually.

Important paths used in the notebook:

```python
CAPTIONS_CSV = "/content/drive/MyDrive/captions.csv"
CUI_VOCAB_PATH = "/content/download_sciebo/cui_to_name_synth_2026.csv"
TRAIN_CONCEPTS_PATH = "/content/download_sciebo/manuel_concepts/train_concepts_manual.csv"
VALID_CONCEPTS_PATH = "/content/download_sciebo/manuel_concepts/valid_concepts_manual.csv"
TRAIN_IMG_DIR = "/content/dataset/train_images"
VALID_IMG_DIR = "/content/dataset/valid_images"
```

### 3. Run the notebook

Open:

```text
imageclef.ipynb
```

Run the notebook sections in order:

1. Data extraction and split
2. Dataset loading
3. Model initialization
4. Training or checkpoint loading
5. Caption generation
6. Submission export

---

## Reproducibility Notes

This repository currently keeps the full workflow in a notebook.  
For stronger reproducibility, the following files should be added:

```text
requirements.txt
config.yaml
src/
├── dataset.py
├── model.py
├── train.py
├── infer.py
└── evaluate.py
```

Recommended improvements:

- move hard-coded Colab paths into a config file,
- add random seed control,
- save all metric outputs to `results.json`,
- include a small sample of generated captions,
- add exact checkpoint names used for final submission,
- and provide a clean inference-only script.

---

## Limitations

- The dataset and model checkpoints are not included.
- The code is notebook-based and depends on Colab/Google Drive paths.
- The final captions contain some tokenization artifacts and incomplete medical words.
- The model uses a relatively small language decoder (`distilgpt2`), which limits caption fluency.
- More specialized medical vision-language models may improve both factuality and fluency.

---

## Future Work

- Replace DistilGPT-2 with a stronger medical or biomedical language decoder.
- Use a pretrained medical vision-language model such as BLIP-style or Med-VLM architecture.
- Add constrained decoding using predicted CUIs.
- Improve post-processing to fix broken medical terms.
- Add ensemble inference across checkpoints.
- Package the notebook into reusable Python modules.
- Release a clean inference script for reproducing the final CSV.

---

## Disclaimer

This repository is for research and benchmark participation in medical image captioning.  
The generated captions are not intended for clinical use or medical decision-making.
