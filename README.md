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
