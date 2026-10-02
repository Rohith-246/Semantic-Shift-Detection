import json
import os
import re
from collections import Counter
from typing import List, Tuple
import urllib.error
import urllib.request

import gradio as gr
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Optional fallback key in code (or enter it directly into the UI)
API_KEY_FALLBACK = ""

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "i", "in",
    "is", "it", "its", "of", "on", "or", "our", "that", "the", "their",
    "them", "they", "this", "to", "was", "we", "were", "which", "with",
    "you", "your", "no", "not", "so", "can", "could", "would", "do", "did",
}

# -------------------------------------------------------------
# 1. CORE ENCODER ARCHITECTURE
# -------------------------------------------------------------
class SemanticShiftEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    def forward(self, encodings, target_indices):
        outputs = self.encoder(**encodings)
        # Summation pooling across layers 9-12
        hidden_states = torch.stack(outputs.hidden_states[-4:], dim=0).sum(dim=0)

        embeddings = []
        for i, idx in enumerate(target_indices):
            if idx == -1:
                embeddings.append(hidden_states[i, 0])
            else:
                embeddings.append(hidden_states[i, idx])
        return torch.stack(embeddings)


def find_target_token_index(tokenizer, sentence: str, target_word: str) -> int:
    base_word = target_word.split("_")[0].lower().strip()
    clean_sentence = sentence.lower()

    match = re.search(r"\b" + re.escape(base_word) + r"\b", clean_sentence)
    if not match:
        match = re.search(re.escape(base_word), clean_sentence)
        if not match:
            return -1

    start_char, end_char = match.start(), match.end()
    encoding = tokenizer(
        sentence,
        return_offsets_mapping=True,
        truncation=True,
        return_tensors="pt",
    )
    offsets = encoding.offset_mapping[0]

    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start <= start_char < tok_end or tok_start < end_char <= tok_end:
            return idx
    return -1


def make_batch(tokenizer, sentences: List[str], target_word: str, device="cpu"):
    plain_sentences = [" ".join(w.split("_")[0] for w in s.split()) for s in sentences]
    encodings = tokenizer(plain_sentences, padding=True, truncation=True, return_tensors="pt")
    target_indices = [find_target_token_index(tokenizer, s, target_word) for s in sentences]
    encodings = {k: v.to(device) for k, v in encodings.items()}
    return encodings, target_indices


# -------------------------------------------------------------
# 2. DISTRIBUTIONAL CENTROIDS & WASSERSTEIN DISTANCE
# -------------------------------------------------------------
def get_sentence_embeddings(model, tokenizer, sentences: List[str], target_word: str, device="cpu"):
    encodings, target_indices = make_batch(tokenizer, sentences, target_word, device)
    embeddings = model(encodings, target_indices)
    return F.normalize(embeddings, p=2, dim=1)


def compute_distributional_distance(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    pts1 = emb1.detach().cpu().numpy()
    pts2 = emb2.detach().cpu().numpy()

    k1 = min(2, len(pts1))
    k2 = min(2, len(pts2))

    c1 = KMeans(n_clusters=k1, n_init=5, random_state=42).fit(pts1).cluster_centers_ if len(pts1) >= 2 else pts1
    c2 = KMeans(n_clusters=k2, n_init=5, random_state=42).fit(pts2).cluster_centers_ if len(pts2) >= 2 else pts2

    dist_matrix = cdist(c1, c2, metric="cosine")
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    return float(dist_matrix[row_ind, col_ind].mean())


def extract_salient_terms(sentences: List[str], target_word: str, limit: int = 6) -> List[str]:
    target = target_word.split("_")[0].lower()
    tokens = []
    for sentence in sentences:
        words = re.findall(r"[a-zA-Z]+", sentence.lower())
        tokens.extend(
            word for word in words
            if word != target and word not in STOP_WORDS and len(word) > 2
        )
    return [word for word, _ in Counter(tokens).most_common(limit)]


def format_chips_html(terms: List[str], color_hex: str) -> str:
    if not terms:
        return "<span style='color: #888;'>None detected</span>"
    return " ".join([
        f"<span style='background-color: {color_hex}; color: white; padding: 4px 10px; "
        f"border-radius: 12px; font-size: 13px; font-weight: 500; display: inline-block; margin: 3px;'>{t}</span>"
        for t in terms
    ])


# -------------------------------------------------------------
# 3. HIGH-PERFORMANCE SENSE SYNTHESIZER
# -------------------------------------------------------------
def synthesize_meanings(
    word: str,
    s1_list: List[str],
    s2_list: List[str],
    s1_terms: List[str],
    s2_terms: List[str],
    distance: float,
    user_key: str = "",
) -> Tuple[str, str, str]:
    api_key = user_key.strip() or os.environ.get("GEMINI_API_KEY", "").strip() or API_KEY_FALLBACK.strip()

    if not api_key:
        print("[Sense Synthesizer] Notice: No API key provided. Using anchor profile.")
        p1 = f"Canonical usage anchored by context tokens: [{', '.join(s1_terms)}]."
        p2 = f"Modern usage anchored by context tokens: [{', '.join(s2_terms)}]."
        evo = f"Wasserstein distributional shift metric is {distance:.4f}. Provide a Gemini API Key to synthesize natural-language meanings."
        return p1, p2, evo

    prompt = (
        f"You are a lexicographer analyzing semantic variation for the word: '{word}'.\n\n"
        f"Period 1 Context:\n" + "\n".join(f"- {s}" for s in s1_list[:3]) + "\n\n"
        f"Period 2 Context:\n" + "\n".join(f"- {s}" for s in s2_list[:3]) + "\n\n"
        "Provide:\n"
        "1. Period 1 Meaning: A concise 1-sentence dictionary definition of how the word is used in Period 1.\n"
        "2. Period 2 Meaning: A concise 1-sentence dictionary definition of how the word is used in Period 2.\n"
        "3. Semantic Evolution: A concise 1-2 sentence explanation of how the meaning transitioned between Period 1 and Period 2.\n\n"
        "Format strictly as:\n"
        "Period 1 Meaning: <definition>\n"
        "Period 2 Meaning: <definition>\n"
        "Semantic Evolution: <evolution summary>"
    )

    text = ""
    candidate_models = ["gemini-3.8-flash", "gemini-2.0-flash", "gemini-2.5-flash"]

    # Try Google GenAI SDK
    if GENAI_AVAILABLE:
        try:
            client = genai.Client(api_key=api_key)
            for model_id in candidate_models:
                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=prompt,
                    )
                    if response and response.text:
                        text = response.text.strip()
                        break
                except Exception:
                    continue
        except Exception as e:
            print(f"[Sense Synthesizer] SDK initialization note: {e}")

    # Fallback to direct REST if SDK did not return text
    if not text:
        headers = {"Content-Type": "application/json"}
        payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
        for model_id in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
            try:
                req = urllib.request.Request(url, data=payload, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    text = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    if text:
                        break
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    print(f"[Sense Synthesizer] REST HTTP Error {e.code}: {e.read().decode('utf-8')[:200]}")
            except Exception as e:
                print(f"[Sense Synthesizer] REST connection issue: {e}")

    # Parse response
    if text:
        p1_match = re.search(r"Period 1 Meaning:\s*(.+)", text, re.IGNORECASE)
        p2_match = re.search(r"Period 2 Meaning:\s*(.+)", text, re.IGNORECASE)
        evo_match = re.search(r"Semantic Evolution:\s*(.+)", text, re.IGNORECASE | re.DOTALL)

        p1_def = p1_match.group(1).strip().split("\n")[0] if p1_match else ""
        p2_def = p2_match.group(1).strip().split("\n")[0] if p2_match else ""
        evo_def = evo_match.group(1).strip() if evo_match else ""

        if not p1_def or not p2_def:
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if len(lines) >= 3:
                p1_def = lines[0]
                p2_def = lines[1]
                evo_def = " ".join(lines[2:])
            elif len(lines) == 2:
                p1_def = lines[0]
                p2_def = lines[1]
                evo_def = f"Semantic shift observed across periods with divergence {distance:.4f}."
            else:
                p1_def, p2_def, evo_def = text, text, f"Divergence: {distance:.4f}."

        return p1_def, p2_def, evo_def

    # Default fallback
    p1 = f"Canonical usage anchored by context tokens: [{', '.join(s1_terms)}]."
    p2 = f"Modern usage anchored by context tokens: [{', '.join(s2_terms)}]."
    evo = f"Wasserstein distributional shift is {distance:.4f}. (API connection unavailable)."
    return p1, p2, evo


# -------------------------------------------------------------
# 4. INFERENCE PIPELINE
# -------------------------------------------------------------
def analyze_input(word: str, old_sentences: str, new_sentences: str, api_key_input: str):
    old_sent_list = [s.strip() for s in old_sentences.split("\n") if s.strip()]
    new_sent_list = [s.strip() for s in new_sentences.split("\n") if s.strip()]

    if not old_sent_list or not new_sent_list:
        error_html = "<div style='color: red; font-weight: bold;'>Error: Enter sentences for both periods.</div>"
        return error_html, "0.0000", "0.0%", "", "", "", ""

    model.eval()
    with torch.no_grad():
        emb1 = get_sentence_embeddings(model, tokenizer, old_sent_list, word, device)
        emb2 = get_sentence_embeddings(model, tokenizer, new_sent_list, word, device)
        distance = compute_distributional_distance(emb1, emb2)

    # 3-Band Calibrated Scale
    if distance < 0.35:
        verdict_html = (
            "<div style='background-color: #ecfdf5; border-left: 6px solid #10b981; "
            "padding: 12px; border-radius: 6px; color: #065f46; font-weight: bold; font-size: 16px;'>"
            "✅ STABLE / NO SHIFT</div>"
        )
        margin = abs(0.35 - distance)
    elif distance < 0.55:
        verdict_html = (
            "<div style='background-color: #fffbeb; border-left: 6px solid #f59e0b; "
            "padding: 12px; border-radius: 6px; color: #b45309; font-weight: bold; font-size: 16px;'>"
            "⚠️ DOMAIN / EMERGING SHIFT</div>"
        )
        margin = min(abs(distance - 0.35), abs(0.55 - distance))
    else:
        verdict_html = (
            "<div style='background-color: #fee2e2; border-left: 6px solid #ef4444; "
            "padding: 12px; border-radius: 6px; color: #991b1b; font-weight: bold; font-size: 16px;'>"
            "🚨 RADICAL SEMANTIC SHIFT</div>"
        )
        margin = abs(distance - 0.55)

    confidence = 2.0 * (1.0 / (1.0 + torch.exp(-torch.tensor(16.0 * margin))) - 0.5).item()

    era1_terms = extract_salient_terms(old_sent_list, word)
    era2_terms = extract_salient_terms(new_sent_list, word)

    p1_meaning, p2_meaning, evo_summary = synthesize_meanings(
        word, old_sent_list, new_sent_list, era1_terms, era2_terms, distance, api_key_input
    )

    badges_html = (
        f"<div style='margin-top: 10px;'>"
        f"<div style='margin-bottom: 6px;'><b>Era 1 Salient Features:</b> {format_chips_html(era1_terms, '#2563eb')}</div>"
        f"<div><b>Era 2 Salient Features:</b> {format_chips_html(era2_terms, '#7c3aed')}</div>"
        f"</div>"
    )

    return (
        verdict_html,
        f"{distance:.4f}",
        f"{confidence * 100:.1f}%",
        p1_meaning,
        p2_meaning,
        evo_summary,
        badges_html,
    )


# -------------------------------------------------------------
# 5. GRADIO UI
# -------------------------------------------------------------
def build_interface():
    with gr.Blocks(title="Semantic Shift Detection") as demo:
        gr.Markdown(
            "# 🔬 Semantic Shift Detection\n"
            "Distributional semantic divergence detection powered by fine-tuned BERT "
            "representations (Layers 9–12) and Wasserstein Optimal Transport."
        )

        with gr.Accordion("⚙️ Optional Configuration: Gemini API Key", open=False):
            api_key_input = gr.Textbox(
                label="Gemini API Key (Leave blank to use system environment key)",
                placeholder="Paste API key here (e.g., AIzaSy...)",
                type="password",
                value="",
            )

        with gr.Row():
            word_input = gr.Textbox(
                label="Target Word",
                placeholder="e.g., apple, cap, record, cell, tree",
                value="apple",
            )

        with gr.Row():
            with gr.Column():
                old_sentences = gr.Textbox(
                    label="Historical / Canonical Era (Period 1)",
                    placeholder="Enter sentences (one per line)",
                    lines=5,
                    value="She picked a crisp red apple directly from the garden branch.\n"
                          "He baked a warm cinnamon pie filled with sliced green apple.\n"
                          "The basket was brimming with fresh cider and sweet ripe apple.",
                )
            with gr.Column():
                new_sentences = gr.Textbox(
                    label="Modern / Shifted Era (Period 2)",
                    placeholder="Enter sentences (one per line)",
                    lines=5,
                    value="The company released a new operating system update for every apple smartphone.\n"
                          "He bought shares of apple stock before the Silicon Valley product launch.\n"
                          "Her laptop was serviced at the downtown apple store yesterday.",
                )

        submit_btn = gr.Button("Analyze Shift", variant="primary")

        gr.Markdown("### 📊 Quantitative Diagnostics")
        with gr.Row():
            verdict_box = gr.HTML()
            distance_box = gr.Textbox(label="Wasserstein Metric (Divergence)", interactive=False)
            confidence_box = gr.Textbox(label="Confidence Interval", interactive=False)

        gr.Markdown("### 📖 Semantic Meaning & Sense Evolution")
        with gr.Row():
            meaning_p1 = gr.Textbox(label="Period 1 Synthesized Meaning (Canonical)", lines=3, interactive=False)
            meaning_p2 = gr.Textbox(label="Period 2 Synthesized Meaning (Modern)", lines=3, interactive=False)

        evolution_box = gr.Textbox(
            label="Semantic Evolution & Shift Explanation",
            lines=3,
            interactive=False,
            placeholder="Natural-language sense evolution narrative will appear here..."
        )

        gr.Markdown("### 🏷️ Salient Context Anchors")
        context_badges_box = gr.HTML()

        submit_btn.click(
            fn=analyze_input,
            inputs=[word_input, old_sentences, new_sentences, api_key_input],
            outputs=[
                verdict_box,
                distance_box,
                confidence_box,
                meaning_p1,
                meaning_p2,
                evolution_box,
                context_badges_box,
            ],
        )

    return demo


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting engine on {device}...")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = SemanticShiftEncoder("bert-base-uncased").to(device)

    weights_path = os.path.join(
        BASE_DIR, "data", "semeval2020_eng", "semantic_encoder_full.pt"
    )
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Loaded calibrated semantic_encoder_full.pt weights.")
    else:
        print("Calibrated checkpoint not found; running on base BERT.")

    custom_css = ".gradio-container { max-width: 950px !important; margin: auto; }"
    demo = build_interface()
    demo.launch(css=custom_css)
