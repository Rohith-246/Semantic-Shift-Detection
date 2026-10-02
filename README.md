---
title: Semantic Shift Detection
emoji: chart_with_upwards_trend
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
---

# Semantic Shift Detection

A BERT-based semantic shift detector for comparing how a target word is used across two time periods. The project uses contextual embeddings and contrastive learning, then provides an interactive Gradio interface with an explanation of the likely meanings in each period.

## Features

- Train a semantic shift encoder with the SemEval-2020 Task 1 English dataset
- Compare old-period and new-period sentence contexts
- Return an embedding distance, prediction, and confidence score
- Explain likely meanings using WordNet and context terms
- Generate a validation report
- Launch a local or public Gradio interface

## Requirements

- Python 3.10 or newer
- Internet access for the pretrained BERT model and dataset download
- At least several GB of free disk space for model and dataset files

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
python3 -m nltk.downloader wordnet omw-1.4
```

## Usage

Run commands from the repository directory:

```bash
cd /path/to/Semantic-Shift-Detection
```

### Train the model

```bash
python3 "Semantic Shift Detection.py" train
```

Training creates these local files:

- `semantic_shift_model.pt`
- `val_words.txt`

### Generate an evaluation report

```bash
python3 "Semantic Shift Detection.py" report
```

The report is written to `evaluation_report.md`.

### Launch Gradio

```bash
python3 "Semantic Shift Detection.py"
```

The script prints a local URL and a temporary public `gradio.live` URL. Enter a target word and provide one or more sentences for each period. The output includes the prediction, distance, confidence, likely old and new meanings, and context evidence.

You can also launch explicitly:

```bash
python3 "Semantic Shift Detection.py" gradio
```

## Example

Target word:

```text
bank
```

Old-period sentences:

```text
The bank approved the loan.
Money in the bank earns interest.
```

New-period sentences:

```text
The river bank flooded.
They walked along the bank of the river.
```

## Project Notes

The trained model and downloaded dataset are intentionally excluded from Git because they are large generated artifacts. They are downloaded or created locally when needed. The decision threshold is currently set to `0.4437`, based on the existing validation workflow.

# 🔬 Neuro-Symbolic Semantic Shift Detection Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://semantic-shift-detection-cs9htc5jqjjznwuzjej5nv.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30+-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> An end-to-end lexical semantic change detection system combining fine-tuned transformer representations, optimal transport geometric metrics, and LLM-driven lexicographic sense synthesis. Calibrated against the SemEval-2020 Task 1 benchmark.

---

## 📌 Overview

Words naturally alter, expand, or invert their meanings across time and socio-technical domains (e.g., *apple* shifting from botanical fruit to consumer technology ecosystem; *cap* shifting from headwear to modern vernacular for deception).

Traditional distributional approaches either rely on static word embeddings (Word2Vec, fastText) that conflate polysemy, or pure LLM prompt heuristics that lack rigorous mathematical grounding. 

This engine implements a hybrid **neuro-symbolic architecture**:
1. **Geometric Distributional Metric**: Extracts contextualized token embeddings across historical and contemporary usage distributions via fine-tuned BERT layer summation, computing continuous drift using **Wasserstein Optimal Transport**.
2. **Lexicographic Sense Synthesis**: Grounded natural-language sense definitions and historical evolution narratives dynamically synthesized via Google's Gemini models with resilient fallback routing.

---

## 🏛️ System Architecture

```text
               Target Word + Dual-Corpus Contexts
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
   Period 1 Sentences                    Period 2 Sentences
            │                                     │
   ┌─────────────────┐                   ┌─────────────────┐
   │ BERT Tokenizer  │                   │ BERT Tokenizer  │
   │ & Target Index  │                   │ & Target Index  │
   └────────┬────────┘                   └────────┬────────┘
            ▼                                     ▼
 ┌─────────────────────┐               ┌─────────────────────┐
 │ Fine-Tuned BERT Core│               │ Fine-Tuned BERT Core│
 │ Layers 9-12 Pooling │               │ Layers 9-12 Pooling │
 └──────────┬──────────┘               └──────────┬──────────┘
            │                                     │
            ▼                                     ▼
    Cluster Centers μ₁                    Cluster Centers μ₂
            │                                     │
            └──────────────────┬──────────────────┘
                               ▼
                Wasserstein Optimal Transport
               (Cosine Metric + Hungarian Matching)
                               │
                               ▼
                   Calibrated Decision Scale
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
          D < 0.35      0.35 ≤ D < 0.55     D ≥ 0.55
          [STABLE]          [DOMAIN]        [RADICAL]
                               │
                               ▼
                Salient Lexical Anchor Extraction
                               │
                               ▼
                  Gemini LLM Sense Synthesis
         (Dictionary Definitions + Shift Evolution Summary)

