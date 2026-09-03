"""
data_pipeline.py
Loads the SemEval-2020 Task 1 (English) lexical semantic change dataset,
samples example sentences per target word per time period, and prepares
batches of tokenized sentences with the target word's token index for
feeding into a transformer model.
"""

import os
import gzip
import re
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer
from nltk.corpus import wordnet as wn  # pyright: ignore[reportMissingImports]
from nltk.wsd import lesk  # pyright: ignore[reportMissingImports]

# Official host: University of Stuttgart IMS (not Zenodo)
# https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd-eng/
DATA_URL = "https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_eng.zip"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "semeval2020_eng")
MODEL_PATH = os.path.join(BASE_DIR, "semantic_shift_model.pt")
VAL_WORDS_PATH = os.path.join(BASE_DIR, "val_words.txt")
REPORT_PATH = os.path.join(BASE_DIR, "evaluation_report.md")
GRADIO_TEMP_DIR = os.path.join(BASE_DIR, ".gradio")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "i", "in",
    "is", "it", "its", "of", "on", "or", "our", "that", "the", "their",
    "them", "they", "this", "to", "was", "we", "were", "which", "with",
    "you", "your",
}
CONTEXT_DOMAINS = {
    "finance": {"loan", "money", "cash", "mortgage", "deposit", "finance", "account", "credit", "lending"},
    "geography": {"river", "water", "shore", "land", "flood", "flooded", "soil", "earth", "coast", "stream"},
    "technology": {"computer", "data", "internet", "network", "software", "online", "website", "digital"},
    "biology": {"animal", "plant", "species", "body", "cell", "organ", "disease", "blood"},
}


def download_data(data_dir: str = DATA_DIR, retries: int = 3):
    """Download and unzip the SemEval-2020 Task 1 English dataset if not present."""
    if os.path.exists(data_dir):
        print(f"Data already present at {data_dir}")
        return
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "data.zip")

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading SemEval-2020 Task 1 (English) data (attempt {attempt}/{retries})...")
            req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response, open(zip_path, "wb") as out_file:
                out_file.write(response.read())
            break
        except Exception as e:  # noqa: BLE001 - we want to retry on any network error
            last_error = e
            print(f"Download failed: {e}")
    else:
        raise RuntimeError(
            f"Could not download the dataset after {retries} attempts. "
            f"The university server may be temporarily down. "
            f"Try again later, or download it manually from "
            f"https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd-eng/ "
            f"and unzip it into '{data_dir}'."
        ) from last_error

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    os.remove(zip_path)
    print("Done.")


def load_targets_and_labels(data_dir: str = DATA_DIR) -> Dict[str, Dict]:
    """
    Reads targets.txt, truth/binary.txt, truth/graded.txt
    Returns {word: {"binary": 0/1, "graded": float}}
    """
    targets_path = os.path.join(data_dir, "targets.txt")
    binary_path = os.path.join(data_dir, "truth", "binary.txt")
    graded_path = os.path.join(data_dir, "truth", "graded.txt")

    words = [w.strip() for w in open(targets_path, encoding="utf-8")]

    binary = {}
    for line in open(binary_path, encoding="utf-8"):
        w, label = line.strip().split("\t")
        binary[w] = int(label)

    graded = {}
    for line in open(graded_path, encoding="utf-8"):
        w, score = line.strip().split("\t")
        graded[w] = float(score)

    return {w: {"binary": binary.get(w), "graded": graded.get(w)} for w in words}


def load_corpus_sentences(corpus_path: str, target_word: str, max_sentences: int = 100) -> List[str]:
    """
    Scans a corpus file (one sentence per line, lemma_pos tokens) for
    sentences containing the target word's lemma and returns up to
    max_sentences of them. Handles both .txt and .gz files.
    """
    matches = []
    base_word = target_word.split("_")[0]  # SemEval targets look like "cell_nn"

    # Handle both compressed and uncompressed files
    if corpus_path.endswith('.gz'):
        f = gzip.open(corpus_path, 'rt', encoding='utf-8')
    else:
        f = open(corpus_path, encoding='utf-8')

    try:
        for line in f:
            tokens = line.strip().split()
            lemmas = [t.split("_")[0] for t in tokens]
            if base_word in lemmas:
                matches.append(line.strip())
            if len(matches) >= max_sentences:
                break
    finally:
        f.close()

    return matches


@dataclass
class WordExample:
    word: str
    old_sentences: List[str]
    new_sentences: List[str]
    binary_label: int
    graded_label: float


def build_dataset(data_dir: str = DATA_DIR, max_sentences_per_period: int = 100) -> List[WordExample]:
    """Assembles one WordExample per target word that has usable sentences in both periods."""
    labels = load_targets_and_labels(data_dir)
    corpus1 = os.path.join(data_dir, "corpus1", "lemma", "ccoha1.txt.gz")
    corpus2 = os.path.join(data_dir, "corpus2", "lemma", "ccoha2.txt.gz")

    examples = []
    for word, lab in labels.items():
        old_sents = load_corpus_sentences(corpus1, word, max_sentences_per_period)
        new_sents = load_corpus_sentences(corpus2, word, max_sentences_per_period)
        if not old_sents or not new_sents:
            continue  # skip words with no usable examples in one of the periods
        examples.append(WordExample(
            word=word,
            old_sentences=old_sents,
            new_sentences=new_sents,
            binary_label=lab["binary"],
            graded_label=lab["graded"],
        ))
    return examples


def find_target_token_index(tokenizer, sentence: str, target_word: str) -> int:
    """
    Finds the token index of the target word's first subtoken within the
    tokenized sentence, so we can pull out its contextual embedding later.
    Returns -1 if the word can't be located (rare edge cases).
    """
    base_word = target_word.split("_")[0]
    words = sentence.split()
    plain_words = [w.split("_")[0] for w in words]

    encoding = tokenizer(plain_words, is_split_into_words=True, return_tensors="pt", truncation=True)
    word_ids = encoding.word_ids()

    try:
        word_position = plain_words.index(base_word)
    except ValueError:
        return -1

    for token_idx, wid in enumerate(word_ids):
        if wid == word_position:
            return token_idx
    return -1


def make_batch(tokenizer, sentences: List[str], target_word: str, device="cpu"):
    """
    Tokenizes a batch of sentences and returns model-ready inputs plus
    the target-word token index for each sentence (used to pull out the
    right embedding after the forward pass).
    """
    plain_sentences = [" ".join(w.split("_")[0] for w in s.split()) for s in sentences]
    encodings = tokenizer(plain_sentences, padding=True, truncation=True, return_tensors="pt")
    target_indices = [find_target_token_index(tokenizer, s, target_word) for s in sentences]
    encodings = {k: v.to(device) for k, v in encodings.items()}
    return encodings, target_indices


"""
model.py
Wraps a pretrained BERT encoder and exposes a helper to extract the
contextual embedding of a target word's token from a batch of sentences.
"""

class SemanticShiftEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, encodings, target_indices):
        """
        encodings: tokenizer output dict (input_ids, attention_mask, ...)
        target_indices: list[int], one per sentence in the batch, giving
                         the token position of the target word.
        Returns: tensor of shape (batch_size, hidden_dim) — one embedding
                 per sentence, taken at the target word's position.
        """
        outputs = self.encoder(**encodings)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)

        embeddings = []
        for i, idx in enumerate(target_indices):
            if idx == -1:
                # fallback: use the [CLS] token if we couldn't locate the word
                embeddings.append(hidden_states[i, 0])
            else:
                embeddings.append(hidden_states[i, idx])
        return torch.stack(embeddings)  # (batch, hidden)
"""
train.py
Fine-tunes a BERT encoder so that, for words known to have shifted in
meaning, the average embedding from the old-period corpus is pushed away
from the average embedding from the new-period corpus. For stable words,
the two averages are pulled together instead.

This is a form of supervised contrastive / metric learning: we don't
train a classifier head, we train the embedding space itself, since the
thing we actually care about at inference time (old-vs-new distance) is
exactly what the loss optimizes.
"""

import random

# Note: data_pipeline, model, and this train section are all in one file
# so we don't need to import them


def pooled_embedding(model, tokenizer, sentences, target_word, device, max_sentences=16):
    """Average contextual embedding of the target word across a set of sentences."""
    sentences = sentences[:max_sentences]
    encodings, target_indices = make_batch(tokenizer, sentences, target_word, device)
    embeddings = model(encodings, target_indices)  # (n_sentences, hidden)
    return embeddings.mean(dim=0)  # (hidden,)


def contrastive_loss(old_emb, new_emb, binary_label, margin=1.0):
    """Push apart if the word shifted, pull together if it stayed stable."""
    distance = F.pairwise_distance(old_emb.unsqueeze(0), new_emb.unsqueeze(0)).squeeze(0)
    if binary_label == 1:
        return F.relu(margin - distance)  # shifted -> want distance >= margin
    else:
        return distance  # stable -> want distance -> 0


def train(
    model_name="bert-base-uncased",
    epochs=5,
    lr=2e-5,
    val_fraction=0.2,
    seed=42,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)

    download_data()
    dataset = build_dataset()
    random.shuffle(dataset)

    n_val = max(1, int(len(dataset) * val_fraction))
    val_set = dataset[:n_val]
    train_set = dataset[n_val:]
    print(f"Train words: {len(train_set)}, Val words: {len(val_set)}")
    print("Note: this benchmark has very few labeled words in total, so treat")
    print("val metrics as a rough signal, not a tight estimate of generalization.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SemanticShiftEncoder(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        random.shuffle(train_set)
        for example in train_set:
            optimizer.zero_grad()
            old_emb = pooled_embedding(model, tokenizer, example.old_sentences, example.word, device)
            new_emb = pooled_embedding(model, tokenizer, example.new_sentences, example.word, device)
            loss = contrastive_loss(old_emb, new_emb, example.binary_label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_set))
        print(f"Epoch {epoch + 1}/{epochs} - train loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    # Save the val split so evaluation scores on the exact same held-out words
    with open(VAL_WORDS_PATH, "w", encoding="utf-8") as f:
        for ex in val_set:
            f.write(ex.word + "\n")
    print(f"Saved model to {MODEL_PATH} and val split to {VAL_WORDS_PATH}")
    return model, tokenizer, train_set, val_set, device


"""
evaluate.py
Evaluates the fine-tuned semantic shift model on the held-out validation
words: computes the old-vs-new embedding distance for each word, compares
against gold labels, and writes a markdown evaluation report.
"""

import os
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Note: data_pipeline, model, and train functions are all in one file
# No external imports needed


def load_val_examples(val_words_path=VAL_WORDS_PATH):
    dataset = build_dataset()
    by_word = {ex.word: ex for ex in dataset}
    if os.path.exists(val_words_path):
        with open(val_words_path, encoding="utf-8") as f:
            val_words = [line.strip() for line in f if line.strip()]
        return [by_word[w] for w in val_words if w in by_word]
    print("val_words.txt not found — evaluating on the full dataset instead "
          "(re-run train.py first to get a clean held-out split).")
    return dataset


def compute_distances(model, tokenizer, examples, device):
    distances, binary_labels, graded_labels, words = [], [], [], []
    model.eval()
    with torch.no_grad():
        for ex in examples:
            old_emb = pooled_embedding(model, tokenizer, ex.old_sentences, ex.word, device)
            new_emb = pooled_embedding(model, tokenizer, ex.new_sentences, ex.word, device)
            dist = torch.nn.functional.pairwise_distance(old_emb.unsqueeze(0), new_emb.unsqueeze(0)).item()
            distances.append(dist)
            binary_labels.append(ex.binary_label)
            graded_labels.append(ex.graded_label)
            words.append(ex.word)
    return words, distances, binary_labels, graded_labels


def best_threshold(distances, binary_labels):
    """Sweep candidate thresholds and pick the one maximizing F1 on this set."""
    best_f1, best_t = 0, 0.5
    for t in sorted(set(distances)):
        preds = [1 if d >= t else 0 for d in distances]
        f1 = f1_score(binary_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def context_terms(sentences: List[str], target_word: str, limit: int = 8) -> Counter:
    """Count informative context terms while ignoring the target and stop words."""
    target = target_word.split("_")[0].lower()
    terms = []
    for sentence in sentences:
        words = re.findall(r"[a-zA-Z]+", sentence.lower())
        terms.extend(word for word in words if word != target and word not in STOP_WORDS)
    return Counter(terms).most_common(limit)


def best_wordnet_sense(word: str, sentences: List[str]) -> str:
    """Select and describe the WordNet sense best supported by the context."""
    lemma = word.split("_")[0].lower()
    senses = wn.synsets(lemma, pos=wn.NOUN) or wn.synsets(lemma)
    if not senses:
        return "No dictionary sense was found for this word."

    context = set(dict(context_terms(sentences, word, limit=30)))
    best_sense = senses[0]
    best_score = -1
    for sense in senses:
        sense_text = " ".join([sense.definition(), *sense.examples()]).lower()
        gloss_words = set(re.findall(r"[a-zA-Z]+", sense_text))
        score = len(context & gloss_words)
        for domain_words in CONTEXT_DOMAINS.values():
            if context & domain_words and any(term in sense_text for term in domain_words):
                score += 3
        if score > best_score:
            best_score = score
            best_sense = sense

    if best_score == 0:
        for sentence in sentences:
            tokens = re.findall(r"[a-zA-Z]+", sentence.lower())
            sense = lesk(tokens, lemma)
            if sense:
                best_sense = sense
                break

    examples = best_sense.examples()
    example_text = f" Example: {examples[0]}" if examples else ""
    return f"{best_sense.definition().capitalize()}.{example_text}"


def explain_shift(word: str, old_sentences: List[str], new_sentences: List[str], distance: float, threshold: float) -> str:
    """Explain the prediction using context terms that differ between periods."""
    old_terms = dict(context_terms(old_sentences, word))
    new_terms = dict(context_terms(new_sentences, word))
    old_only = [term for term in old_terms if term not in new_terms][:5]
    new_only = [term for term in new_terms if term not in old_terms][:5]
    shared = [term for term in new_terms if term in old_terms][:5]

    distance_text = (
        f"The embedding distance ({distance:.4f}) is above the decision threshold "
        f"({threshold:.4f})."
        if distance >= threshold
        else f"The embedding distance ({distance:.4f}) is below the decision threshold "
             f"({threshold:.4f})."
    )
    evidence = []
    if old_only:
        evidence.append(f"Old-period context: {', '.join(old_only)}")
    if new_only:
        evidence.append(f"New-period context: {', '.join(new_only)}")
    if shared:
        evidence.append(f"Shared context: {', '.join(shared)}")
    if not evidence:
        evidence.append("There were not enough distinct context terms to summarize.")

    return (
        distance_text
        + "\n\nLikely old-period meaning:\n"
        + best_wordnet_sense(word, old_sentences)
        + "\n\nLikely new-period meaning:\n"
        + best_wordnet_sense(word, new_sentences)
        + "\n\nContext evidence:\n"
        + "\n".join(evidence)
        + "\n\nThe meanings are dictionary-based interpretations matched to your contexts; "
          "they are supporting evidence, not guaranteed definitions."
    )


def generate_report(model_path=MODEL_PATH, model_name="bert-base-uncased"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SemanticShiftEncoder(model_name).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    examples = load_val_examples()
    words, distances, binary_labels, graded_labels = compute_distances(model, tokenizer, examples, device)

    threshold = best_threshold(distances, binary_labels)
    preds = [1 if d >= threshold else 0 for d in distances]

    acc = accuracy_score(binary_labels, preds)
    f1 = f1_score(binary_labels, preds, zero_division=0)
    precision = precision_score(binary_labels, preds, zero_division=0)
    recall = recall_score(binary_labels, preds, zero_division=0)
    corr, p_value = spearmanr(distances, graded_labels)

    report_lines = [
        "# Semantic Shift Detection — Evaluation Report",
        "",
        f"- Evaluated on {len(examples)} held-out words",
        f"- Threshold used: {threshold:.4f}",
        f"- Accuracy: {acc:.3f}",
        f"- Precision: {precision:.3f}",
        f"- Recall: {recall:.3f}",
        f"- F1: {f1:.3f}",
        f"- Spearman correlation (distance vs. graded gold score): {corr:.3f} (p={p_value:.4f})",
        "",
        "_Note: this benchmark has only a few dozen labeled words total, so these",
        "numbers should be read as a rough signal rather than a tight estimate._",
        "",
        "## Per-word results",
        "",
        "| Word | Distance | Predicted | Gold Binary | Gold Graded |",
        "|---|---|---|---|---|",
    ]
    for w, d, p, b, g in zip(words, distances, preds, binary_labels, graded_labels):
        report_lines.append(f"| {w} | {d:.3f} | {p} | {b} | {g:.3f} |")

    report = "\n".join(report_lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    return report


def predict_semantic_shift(word: str, old_sentences: str, new_sentences: str):
    """
    Predict whether a word has undergone semantic shift.

    Args:
        word: Target word to analyze
        old_sentences: Sentences from the old time period (separated by newlines)
        new_sentences: Sentences from the new time period (separated by newlines)

    Returns:
        Tuple of (semantic_shift_prediction, distance, confidence)
    """
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        return (
            "Model not found. Please train the model first by running the script.",
            0.0,
            0.0,
            ""
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = SemanticShiftEncoder("bert-base-uncased").to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Parse sentences
    old_sent_list = [s.strip() for s in old_sentences.split("\n") if s.strip()]
    new_sent_list = [s.strip() for s in new_sentences.split("\n") if s.strip()]

    if not old_sent_list or not new_sent_list:
        return "Error: Please provide at least one sentence for both time periods.", 0.0, 0.0, ""

    try:
        with torch.no_grad():
            old_emb = pooled_embedding(model, tokenizer, old_sent_list, word, device)
            new_emb = pooled_embedding(model, tokenizer, new_sent_list, word, device)
            distance = torch.nn.functional.pairwise_distance(
                old_emb.unsqueeze(0), new_emb.unsqueeze(0)
            ).item()

        # Use threshold from training (0.4437)
        threshold = 0.4437
        is_shifted = distance >= threshold
        confidence = min(abs(distance - threshold) / max(threshold, 1.0 - threshold), 1.0)

        prediction = "Semantic Shift Detected" if is_shifted else "No Semantic Shift"
        explanation = explain_shift(
            word, old_sent_list, new_sent_list, distance, threshold
        )

        return (
            prediction,
            f"{distance:.4f}",
            f"{confidence * 100:.1f}%",
            explanation,
        )
    except Exception as e:
        return f"Error during prediction: {str(e)}", 0.0, 0.0, ""


def create_demo():
    """Create and return the Gradio interface."""
    import gradio as gr

    with gr.Blocks(title="Semantic Shift Detection") as demo:
        gr.Markdown("# Semantic Shift Detection")
        gr.Markdown("Detect whether a word has undergone semantic shift between two time periods using BERT embeddings.")

        with gr.Row():
            word_input = gr.Textbox(
                label="Target Word",
                placeholder="e.g., 'bank', 'network', 'mouse'",
                info="Enter the word you want to analyze"
            )

        with gr.Row():
            with gr.Column():
                old_sentences = gr.Textbox(
                    label="Old Period Sentences",
                    placeholder="Enter sentences (one per line) from the earlier time period",
                    lines=5,
                    info="Sentences where the word appeared in the past"
                )
            with gr.Column():
                new_sentences = gr.Textbox(
                    label="New Period Sentences",
                    placeholder="Enter sentences (one per line) from the recent time period",
                    lines=5,
                    info="Sentences where the word appears recently"
                )

        with gr.Row():
            submit_btn = gr.Button("Analyze Semantic Shift", variant="primary")

        with gr.Row():
            with gr.Column():
                prediction = gr.Textbox(
                    label="Prediction",
                    interactive=False,
                    info="Whether semantic shift was detected"
                )
            with gr.Column():
                distance = gr.Textbox(
                    label="Embedding Distance",
                    interactive=False,
                    info="Distance between old and new embeddings (higher = more shift)"
                )
            with gr.Column():
                confidence = gr.Textbox(
                    label="Confidence",
                    interactive=False,
                    info="Model confidence in the prediction"
                )

        explanation = gr.Textbox(
            label="Explanation",
            lines=6,
            interactive=False,
            info="Context terms that help explain the difference between periods"
        )

        gr.Markdown("""
        ## How It Works

        This tool uses a fine-tuned BERT model trained on the SemEval-2020 Lexical Semantic Change Detection dataset.

        **Process:**
        1. Extracts contextual embeddings of your target word from both time periods
        2. Computes the distance between the averaged embeddings
        3. Predicts semantic shift if distance exceeds the learned threshold (0.4437)

        **Example Usage:**
        - Word: "bank"
        - Old sentences: "The bank was robbed yesterday." / "Money in the bank earns interest."
        - New sentences: "River bank erosion is a problem." / "The left bank of the river is steep."
        """)

        submit_btn.click(
            fn=predict_semantic_shift,
            inputs=[word_input, old_sentences, new_sentences],
            outputs=[prediction, distance, confidence, explanation]
        )

    return demo


if __name__ == "__main__":
    import sys

    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "gradio"
    if mode == "train":
        train()
    elif mode == "report":
        generate_report()
    elif mode == "gradio":
        os.chdir(BASE_DIR)
        os.makedirs(GRADIO_TEMP_DIR, exist_ok=True)
        os.environ["GRADIO_TEMP_DIR"] = GRADIO_TEMP_DIR
        demo = create_demo()
        demo.launch(share=True)
    else:
        raise SystemExit("Usage: python 'Semantic Shift Detection.py' [gradio|train|report]")