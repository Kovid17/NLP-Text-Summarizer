# 📝 NLP+DL Text Summarization Project

> **"The project utilizes diverse datasets spanning news, dialogue, social media, and scientific domains to build a robust, domain-adaptive text summarization system."**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-green)](https://streamlit.io)

---

## 🎯 Project Overview

A multi-domain text summarization system using Transformer-based deep learning (BART, T5, Pegasus, LED), trained and evaluated across **7 diverse datasets** in 3 progressive phases.

### Architecture Pipeline
```
Input Datasets → Preprocessing & Tokenization → Seq2Seq Transformer
→ Training (Forward / Loss / Backprop / Weight Update)
→ Inference (Beam Search / Sampling)
→ Evaluation (ROUGE-1/2/L · BLEU · METEOR)
→ Output: Summaries + Reports + Model Artifacts
```

---

## 📊 Dataset → Model Mapping

| Phase | Dataset | Domain | Type | Model |
|-------|---------|--------|------|-------|
| 🟢 1 | **CNN/DailyMail** | News | Article → Multi-sentence | `facebook/bart-large-cnn` |
| 🟢 1 | **SAMSum** | Dialogue | Conversation → Summary | `philschmid/bart-large-cnn-samsum` |
| 🔵 2 | **XSum** | BBC News | Article → One-line | `facebook/bart-large-xsum` |
| 🔵 2 | **Gigaword** | Headlines | Sentence → Headline | `google/pegasus-gigaword` |
| 🟡 3 | **ArXiv** | Scientific | Paper → Abstract | `google/pegasus-arxiv` |
| 🟡 3 | **Reddit TIFU** | Social Media | Story → TL;DR | `facebook/bart-large` (FT) |
| 🔴 3 | **ACLSum** | NLP Research | Paper → Summary | `allenai/led-base-16384` |

---

## 🗂️ Project Structure

```
NLP-Project/
├── requirements.txt          # All dependencies
├── config/
│   ├── model_configs.yaml    # Hyperparameters per model
│   └── dataset_configs.yaml  # Dataset registry
├── src/
│   ├── utils.py              # Logging, seeding, helpers
│   ├── dataset_loader.py     # Unified HF dataset loader
│   ├── preprocessor.py       # Tokenization + DataLoaders
│   ├── model_factory.py      # Load any model by name
│   ├── trainer.py            # Training loop + ROUGE validation
│   ├── evaluator.py          # ROUGE/BLEU/METEOR + plots
│   └── inference.py          # Beam search + sampling
├── data/
│   └── download_datasets.py  # Download all 7 datasets
├── scripts/
│   ├── train.py              # Train any dataset
│   ├── evaluate.py           # Evaluate a checkpoint
│   └── zero_shot_baseline.py # Pre-trained baseline scores
├── notebooks/
│   ├── 01_EDA_CNN_SAMSum.py  # Phase 1 EDA
│   ├── 04_Training_Phase1.py # Phase 1 training walkthrough
│   └── 07_Model_Comparison.py# Cross-model comparison
└── app/
    └── app.py                # Streamlit interactive demo
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
python -m venv venv
venv\Scripts\Activate          # Windows
pip install -r requirements.txt
```

### 2. Download Datasets (Phase 1 first)
```bash
python data/download_datasets.py --phase 1
```

### 3. Run Zero-Shot Baseline (no training needed)
```bash
python scripts/zero_shot_baseline.py --phase 1
```

### 4. Fine-tune a Model
```bash
# CNN/DailyMail with BART
python scripts/train.py --dataset cnn_dailymail --epochs 3

# SAMSum with BART-SAMSum
python scripts/train.py --dataset samsum --epochs 5
```

### 5. Evaluate a Checkpoint
```bash
python scripts/evaluate.py --dataset cnn_dailymail \
    --checkpoint outputs/cnn_dailymail/<run_id>/best_model
```

### 6. Launch the Demo App
```bash
streamlit run app/app.py
```

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **ROUGE-1** | Unigram overlap between prediction and reference |
| **ROUGE-2** | Bigram overlap |
| **ROUGE-L** | Longest Common Subsequence (LCS) |
| **BLEU** | Precision-based n-gram matching |
| **METEOR** | Harmonic mean of precision and recall with synonym matching |

### Expected Baseline Scores (pre-trained, no fine-tuning)

| Dataset | Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---------|-------|---------|---------|---------|
| CNN/DailyMail | BART-Large-CNN | 44.16 | 21.28 | 40.90 |
| SAMSum | BART-SAMSum | 53.22 | 28.97 | 48.60 |
| XSum | BART-Large-XSum | 45.14 | 22.27 | 37.25 |
| Gigaword | Pegasus-Gigaword | 39.12 | 19.86 | 36.24 |
| ArXiv | Pegasus-ArXiv | 44.21 | 17.06 | 38.96 |

---

## 🧠 Model Architectures

### BART (Bidirectional Auto-Regressive Transformer)
- Encoder: Bidirectional transformer (like BERT)
- Decoder: Autoregressive transformer (like GPT)
- Pre-trained with denoising (text corruption + reconstruction)
- **Best for**: News summarization, dialogue

### Pegasus
- Pre-trained with Gap Sentences Generation (GSG)
- Sentences are masked and the model learns to regenerate them
- **Best for**: Abstractive summarization, scientific text

### LED (Longformer Encoder-Decoder)
- Extends BART with sparse attention for long documents
- Global attention on CLS token
- **Best for**: Long document summarization (ArXiv, ACLSum)

---

## 🏗️ Phase-wise Implementation

### Phase 1 — Foundation
- Datasets: CNN/DailyMail + SAMSum
- Goal: Establish baselines, validate pipeline
- Expected time: 2–4 hours on GPU

### Phase 2 — Expansion
- Datasets: XSum + Gigaword
- Goal: One-line & headline summarization
- Expected time: 3–5 hours on GPU

### Phase 3 — Advanced
- Datasets: ArXiv + Reddit TIFU + ACLSum
- Goal: Long documents, noisy text, domain-specific
- Expected time: 6–12 hours on GPU (or use Colab Pro)

---

## 💡 Tips

- **No GPU?** Use `zero_shot_baseline.py` — no training required, uses pre-trained models
- **Google Colab**: Set `device=0` in inference and enable GPU runtime
- **Memory issues**: Reduce `batch_size` and enable `gradient_accumulation_steps: 8` in config
- **Fast experiment**: Set `train_samples: 1000` in `dataset_configs.yaml`

---

## 📄 License
MIT License — feel free to use for academic/research purposes.
