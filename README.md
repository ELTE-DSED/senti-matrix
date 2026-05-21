# SentiMatrix: Parameter-Efficient Fine-Tuning of Encoder-Based Transformers for Multidimensional Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Python · PyTorch · Transformers · HuggingFace · LoRA · AdaLoRA*

> **This work is currently under review at the** ***Machine Learning*** **journal (Springer).**

---

## Description

SentiMatrix is a parameter-efficient fine-tuning framework for transformer-based models across four sentiment analysis tasks: intent-based (binary and multi-class), aspect-based, fine-grained, and emotion detection. The framework benchmarks pretrained models against three adaptation strategies — full fine-tuning, LoRA adapters, and AdaLoRA adapters — enabling direct comparison of predictive performance against computational cost.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage and Reproduction](#usage-and-reproduction)
- [PEFT Methods](#peft-methods)
- [Evaluation Metrics](#evaluation-metrics)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Contact](#contact)

---

## Overview

SentiMatrix addresses challenges in multidimensional sentiment analysis by combining four model adaptation strategies:

- **Pretrained Model Evaluation** — Benchmarking models pretrained on task-specific datasets in a zero-shot setting.
- **Base Model Fine-Tuning** — Full fine-tuning of transformer architectures on target datasets.
- **LoRA Adapter Fine-Tuning** — Applying Low-Rank Adaptation (LoRA) for parameter-efficient adaptation.
- **AdaLoRA Adapter Fine-Tuning** — Applying Adaptive Low-Rank Adaptation (AdaLoRA) for importance-aware parameter budget allocation via SVD-based rank pruning.

---

## Key Results

The tables below summarize model performance across all sentiment analysis tasks. LoRA and AdaLoRA adapters provide competitive accuracy while significantly reducing trainable parameters and GPU memory usage compared to full fine-tuning.

---

### Intent-Based Sentiment Analysis (2-Class)

| Dataset | Model | Accuracy | F1-Score | Parameters | GPU Memory |
|---------|-------|:--------:|:--------:|:----------:|:----------:|
| SST-2 | Pretrained DistilBERT | 98.13% | 98.20% | — | — |
| | Fine-Tuned DistilBERT | 89.09% | 89.51% | 66M | 2.0 GB |
| | LoRA DistilBERT | 89.40% | 89.84% | 758K | 1.0 GB |
| | AdaLoRA DistilBERT | 88.77% | 88.89% | 924K | 1.2 GB |
| SST-2 | Pretrained RoBERTa | 97.51% | 97.60% | — | — |
| | Fine-Tuned RoBERTa | 89.81% | 90.65% | 124M | 3.8 GB |
| | LoRA RoBERTa | 92.00% | 92.21% | 1.2M | 1.2 GB |
| | AdaLoRA RoBERTa | 92.72% | 92.90% | 1.8M | 1.2 GB |
| IMDb | Pretrained BERT | 95.62% | 95.53% | — | — |
| | Fine-Tuned BERT | 94.64% | 94.59% | 109M | 2.7 GB |
| | LoRA BERT | 92.58% | 92.52% | 38K | 1.8 GB |
| | AdaLoRA BERT | 91.38% | 91.41% | 444K | 1.2 GB |

*Pretrained = zero-shot | Fully Fine-Tuned = full parameter update | LoRA = Low-Rank Adapter | AdaLoRA = Adaptive Low-Rank Adapter*

---

### Intent-Based Sentiment Analysis (3-Class)

| Dataset | Model | Accuracy | F1-Score | Parameters | GPU Memory |
|---------|-------|:--------:|:--------:|:----------:|:----------:|
| Twitter | Pretrained RoBERTa | 64.96% | 62.88% | — | — |
| | Fine-Tuned RoBERTa | 81.39% | 81.12% | 125M | 2.9 GB |
| | LoRA BERT | 85.13% | 84.93% | 3.3M | 2.3 GB |
| | AdaLoRA RoBERTa | 75.58% | 75.41% | 1.0M | 1.1 GB |

---

### Aspect-Based Sentiment Analysis (3-Class)

| Domain | Model | Accuracy | F1-Score | Parameters | GPU Memory |
|--------|-------|:--------:|:--------:|:----------:|:----------:|
| Combined (Laptop + Restaurant) | Pretrained DeBERTa | 75.84% | 71.84% | — | — |
| | Fine-Tuned DeBERTa | 78.04% | 66.74% | 184M | 3.5 GB |
| | LoRA DeBERTa | 79.73% | 73.75% | 813K | 2.5 GB |
| | AdaLoRA DeBERTa | 76.35% | 66.72% | 1.2M | 1.7 GB |
| Laptop | Pretrained DeBERTa | 81.90% | 80.71% | — | — |
| | Fine-Tuned DeBERTa | 80.17% | 78.13% | 184M | 4.2 GB |
| | LoRA DeBERTa | 78.02% | 69.84% | 813K | 2.5 GB |
| Restaurant | Pretrained DeBERTa | 73.41% | 68.35% | — | — |
| | Fine-Tuned DeBERTa | 80.06% | 71.14% | 184M | 4.2 GB |
| | LoRA DeBERTa | 79.02% | 68.28% | 813K | 2.4 GB |

> **Note:** AdaLoRA results are reported on the combined Laptop + Restaurant domain only.

---

### Fine-Grained Sentiment Analysis (5-Class / 3-Class)

| Dataset | Model | Accuracy | F1-Score | Parameters | GPU Memory |
|---------|-------|:--------:|:--------:|:----------:|:----------:|
| E-Commerce (5-Class) | Pretrained BERT | 56.87% | 47.82% | — | — |
| | Fine-Tuned BERT | 66.27% | 50.44% | 167M | 3.2 GB |
| | LoRA BERT | 67.73% | 50.31% | 298K | 1.6 GB |
| | AdaLoRA mBERT | 65.30% | 42.01% | 446K | 1.4 GB |
| E-Commerce (3-Class) | Pretrained RoBERTa | 79.12% | 53.51% | — | — |
| | Fine-Tuned RoBERTa | 83.50% | 66.89% | 124M | 2.9 GB |
| | LoRA RoBERTa | 84.96% | 66.47% | 889K | 1.7 GB |
| | AdaLoRA RoBERTa | 84.87% | 66.53% | 1.3M | 3.3 GB |
| Yelp (5-Class) | Pretrained BERT | 55.85% | 55.46% | — | — |
| | Fine-Tuned BERT | 61.13% | 60.44% | 16.7M | 3.7 GB |
| | LoRA BERT | 64.58% | 64.09% | 1.3M | 3.5 GB |
| | AdaLoRA mBERT | 47.53% | 47.75% | 446K | 1.4 GB |
| Yelp (3-Class) | Pretrained RoBERTa | 68.93% | 58.05% | — | — |
| | Fine-Tuned RoBERTa | 82.01% | 77.54% | 124M | 4.2 GB |
| | LoRA RoBERTa | 83.87% | 79.23% | 740K | 4.9 GB |
| | AdaLoRA RoBERTa | 71.66% | 68.48% | 1.3M | 3.3 GB |

---

### Emotion Detection (6-Class)

| Dataset | Model | Accuracy | F1-Score | Parameters | GPU Memory |
|---------|-------|:--------:|:--------:|:----------:|:----------:|
| CARER Emotion | Pretrained DistilBERT | 93.15% | 89.87% | — | — |
| | Fine-Tuned DistilBERT | 93.59% | 90.72% | 66.9M | 2.4 GB |
| | LoRA DistilBERT | 93.40% | 90.61% | 668K | 2.1 GB |
| | AdaLoRA DistilBERT | 93.36% | 93.46% | 927K | 0.5 GB |

*LoRA and AdaLoRA adapters achieve a balance of high accuracy and computational efficiency, updating only a small fraction of total parameters while significantly reducing GPU memory usage.*

---

## Key Findings

- **LoRA achieves competitive performance with minimal parameter updates**, consistently matching fully fine-tuned models while training less than 2% of total parameters across most tasks.
- **AdaLoRA improves upon LoRA through importance-aware rank allocation**, performing competitively on tasks with lower label granularity and delivering strong results particularly in emotion detection and binary classification.
- **Fine-tuning and LoRA significantly improve performance on multi-class and fine-grained tasks**, where pretrained zero-shot models alone are insufficient.
- **Aspect-Based Sentiment Analysis benefits from task-specific adaptation**, particularly in combined domain settings.
- **Both LoRA and AdaLoRA reduce GPU memory consumption** relative to full fine-tuning, making them suitable for resource-efficient and scalable deployment.

---

## Repository Structure

```
SENTI-MATRIX/
│
├── Code/
│   ├── Intent Based Sentiment (2 classes) SST-2 Dataset - Models.ipynb
│   ├── Intent Based Sentiment (2 classes) SST-2 Dataset - Models - AdaLoRA.ipynb
│   ├── Intent Based Sentiment (2 classes) IMDB Dataset.ipynb
│   ├── Intent Based Sentiment (2 classes) IMDB Dataset - AdaLoRA.ipynb
│   ├── Aspect Based Sentiment (3 classes) Laptop + Restaurant ABSA Dataset.ipynb
│   ├── Aspect Based Sentiment (3 classes) Laptop + Restaurant ABSA Dataset - AdaLoRA.ipynb
│   ├── Aspect Based Sentiment (3 classes) Laptop ABSA Dataset.ipynb
│   ├── Aspect Based Sentiment (3 classes) Restaurant ABSA Dataset.ipynb
│   ├── Fine-Grained Sentiment Analysis (3 classes) E-Commerce Reviews.ipynb
│   ├── Fine-Grained Sentiment Analysis (3 classes) E-Commerce Reviews - AdaLoRA.ipynb
│   ├── Fine-Grained Sentiment Analysis (5 classes) E-Commerce Reviews.ipynb
│   ├── Fine-Grained Sentiment Analysis (5 classes) E-Commerce Reviews - AdaLoRA.ipynb
│   ├── Fine-Grained Sentiment Analysis (3 classes) Yelp Reviews.ipynb
│   ├── Fine-Grained Sentiment Analysis (3 classes) Yelp Reviews - AdaLoRA.ipynb
│   ├── Fine-Grained Sentiment Analysis (5 classes) Yelp Reviews.ipynb
│   ├── Fine-Grained Sentiment Analysis (5 classes) Yelp Reviews - AdaLoRA.ipynb
│   ├── Emotion Detection Emotion Dataset.ipynb
│   └── Emotion Detection Emotion Dataset - AdaLoRA.ipynb
│
├── Dataset/
│   ├── E-Commerce Reviews (5 Classes).csv
│   ├── E-Commerce Reviews (3 Classes).csv
│   ├── Emotion Dataset (Small).csv
│   ├── Laptop + Restaurant - ABSA.csv
│   ├── Laptop - ABSA.csv
│   ├── Restaurant - ABSA.csv
│   ├── Twitter Dataset (3 Classes).csv
│   ├── Yelp Reviews (3 Classes).csv
│   ├── Yelp Reviews (5 Classes).csv
│   └── IMDB Dataset (2 Classes).csv
│
└── README.md
```

Each notebook in `Code/` is self-contained and covers one task–dataset combination. Notebooks without an `-AdaLoRA` suffix run the pretrained baseline, full fine-tuning, and LoRA experiments. Notebooks with `-AdaLoRA` suffix run the AdaLoRA variant for the same task.

---

## Installation

### System Requirements

- Python 3.10 or higher
- pip package manager
- NVIDIA GPU recommended for training
- CUDA-enabled environment for GPU acceleration
- 8 GB RAM minimum (16 GB recommended for large models)

### Clone the Repository

```bash
git clone https://github.com/ELTE-DSED/senti-matrix.git
cd senti-matrix
```

### Install Dependencies

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

pip install torch torchvision torchaudio
pip install transformers datasets peft evaluate
pip install pandas scikit-learn matplotlib tqdm
pip install jupyter notebook
```

---

## Datasets

All datasets are pre-processed and stored in the `Dataset/` directory. No additional download steps are required.

| Task | Dataset | Classes | Description |
|:-----|:--------|:-------:|:------------|
| Intent-Based | SST-2 / IMDb | 2 | Binary sentiment (Positive / Negative). |
| Intent-Based | Twitter US Airline | 3 | Multi-class (Positive / Neutral / Negative). |
| Aspect-Based | Laptop / Restaurant (SemEval-2014) | 3 | Sentiment targeting specific aspect entities. |
| Fine-Grained | E-Commerce / Yelp Reviews | 3 & 5 | Granular rating scales (1–5 stars). |
| Emotion Detection | CARER Emotion Dataset | 6 | Sadness, Joy, Love, Anger, Fear, Surprise. |

---

## Usage and Reproduction

All experiments are implemented as Jupyter Notebooks in the `Code/` directory. Each notebook is fully self-contained and follows a three-stage pipeline:

1. **Pretrained Model Evaluation** — Zero-shot benchmarking of task-specific pretrained models from Hugging Face Hub.
2. **Base Model Fine-Tuning** — Full-parameter fine-tuning on the target dataset.
3. **LoRA / AdaLoRA Adapter Training** — Parameter-Efficient Fine-Tuning using LoRA or AdaLoRA.

### Running a Notebook

```bash
jupyter notebook "Code/Emotion Detection Emotion Dataset.ipynb"
```

To reproduce a specific experiment, open the corresponding notebook for that task and dataset and run all cells in order. Each notebook loads its dataset from the `Dataset/` directory, trains the model, and reports accuracy, F1-score, parameter count, and GPU memory usage.

### LoRA Configuration

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    task_type="SEQ_CLS",
    r=16,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
```

### AdaLoRA Configuration

```python
from peft import AdaLoraConfig, get_peft_model

adalora_config = AdaLoraConfig(
    task_type="SEQ_CLS",
    r=16,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(base_model, adalora_config)
model.print_trainable_parameters()
```

---

## PEFT Methods

This study employs two Parameter-Efficient Fine-Tuning strategies: **LoRA** and **AdaLoRA**. Both methods freeze the pre-trained model weights and inject lightweight trainable modules into the Transformer architecture, avoiding full parameter updates.

| Feature | Full Fine-Tuning | LoRA | AdaLoRA |
|:--------|:----------------:|:----:|:-------:|
| Trainable Parameters | 100% | < 2% | < 2% |
| GPU Memory Usage | High | Low | Low |
| Storage per Task | ~500 MB+ | ~5–10 MB | ~5–10 MB |
| Rank Allocation | N/A | Fixed | Adaptive (SVD-based) |
| Performance | Baseline | Competitive / Superior | Competitive / Superior |

**LoRA** injects fixed low-rank decomposition matrices into each Transformer layer. **AdaLoRA** extends this by dynamically pruning and redistributing the parameter budget across weight matrices based on their importance scores via Singular Value Decomposition (SVD), enabling more targeted adaptation.

---

## Evaluation Metrics

Model evaluation is categorized into two dimensions: performance metrics to measure predictive power and efficiency metrics to measure resource utilization.

### Performance Metrics

| Metric | Description |
|:-------|:------------|
| Accuracy | Overall percentage of correct predictions across all classes. |
| Precision | Proportion of positive identifications that were actually correct. |
| Recall | Proportion of actual positives that were correctly identified. |
| F1-Score (Weighted) | Harmonic mean of Precision and Recall, weighted by class frequency. |
| Similarity Score | Semantic closeness between predicted labels and ground truth. |
| Confidence Score | Model probability assigned to its top prediction. |

### Efficiency Metrics

| Metric | Description |
|:-------|:------------|
| Training Time | Total wall-clock time (seconds) required for model convergence. |
| Trainable Parameters | Count of weights updated during training (e.g., ~100M for FFT vs. <1M for LoRA/AdaLoRA). |
| GPU Memory | Peak VRAM consumption during training and inference, measured in GB. |

---

## Reference

This repository accompanies the following article, currently under review:

**Md. Easin Arafat, Muhammad Usman Akmal, Ali S. Abosinnee, and Tamás Orosz.**
"SentiMatrix: Parameter-Efficient Fine-Tuning of Encoder-Based Transformers for Multidimensional Sentiment Analysis."
*Machine Learning*, Springer, 2026. *(Under review)*

```bibtex
@article{arafat2026sentimatrix,
  title     = {SentiMatrix: Parameter-Efficient Fine-Tuning of Encoder-Based Transformers for Multidimensional Sentiment Analysis},
  author    = {Arafat, Md. Easin and
               Akmal, Muhammad Usman and
               Abosinnee, Ali S. and
               Orosz, Tam{\'a}s},
  journal   = {Machine Learning},
  publisher = {Springer},
  year      = {2026},
  note      = {Under review}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Md. Easin Arafat, Muhammad Usman Akmal, Ali S. Abosinnee, Tamás Orosz

---

## Contact

**Md. Easin Arafat**
Doctoral Fellow, Department of Data Science and Engineering
Faculty of Informatics, Eötvös Loránd University (ELTE), Budapest, Hungary
arafatmdeasin@inf.elte.hu

**Muhammad Usman Akmal**
Researcher, Faculty of Informatics, Department of Data Science and Engineering, Eötvös Loránd University (ELTE), Budapest, Hungary
usman.hu1471@gmail.com

**Ali S. Abosinnee**
PhD Candidate, Faculty of Informatics, Department of Data Science and Engineering, Eötvös Loránd University (ELTE), Budapest, Hungary

**Tamás Orosz**
Associate Professor, Faculty of Informatics, Department of Data Science and Engineering, Eötvös Loránd University (ELTE), Budapest, Hungary

---

*Department of Data Science and Engineering, Eötvös Loránd University (ELTE), Budapest, Hungary*
