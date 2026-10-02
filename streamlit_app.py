import json
import os
import re
from collections import Counter
from typing import List, Tuple
import urllib.error
import urllib.request

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Semantic Shift Detection",
    page_icon="🔬",
    layout="wide",
)

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
# 2. CACHED MODEL LOADER
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Loading calibrated BERT model into cloud memory...")
def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = SemanticShiftEncoder("bert-base-uncased").to(device)

    weights_path = os.path.join(BASE_DIR, "data", "semeval2020_eng", "semantic_encoder_full.pt")
    if os.path.exists(weights_path) and os.path.getsize(weights_path) > 1024:
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            status = "Calibrated weights loaded (SemEval-2020 benchmark)."
        except Exception:
            status = "Running on base BERT encoder."
    else:
        status = "Running on base BERT encoder."

    model.eval()
    return tokenizer, model, device, status


# -------------------------------------------------------------
# 3. DISTRIBUTIONAL MATH & ANCHORS
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
# 4. ROBUST SENSE SYNTHESIZER
# -------------------------------------------------------------
def synthesize_meanings(
    word: str,
    s1_list: List[str],
    s2_list: List[str],
    s1_terms: List[str],
    s2_terms: List[str],
    distance: float,
    api_key: str
) -> Tuple[str, str, str, str]:
    if not api_key:
        p1 = f"Anchors: [{', '.join(s1_terms)}]."
        p2 = f"Anchors: [{', '.join(s2_terms)}]."
        evo = f"Wasserstein shift: {distance:.4f}."
        return p1, p2, evo, "No API Key provided. Enter a Gemini API Key to enable sense synthesis."

    prompt = (
        f"You are a lexicographer analyzing semantic variation for the target word: '{word}'.\n\n"
        f"Period 1 Context:\n" + "\n".join(f"- {s}" for s in s1_list[:3]) + "\n\n"
        f"Period 2 Context:\n" + "\n".join(f"- {s}" for s in s2_list[:3]) + "\n\n"
        "Provide:\n"
        "Period 1 Meaning: A concise 1-sentence dictionary definition of how the word is used in Period 1.\n"
        "Period 2 Meaning: A concise 1-sentence dictionary definition of how the word is used in Period 2.\n"
        "Semantic Evolution: A concise 1-2 sentence explanation of how the meaning transitioned between Period 1 and Period 2."
    )

    headers = {"Content-Type": "application/json"}
    payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
    
    candidate_models = ["gemini-2.0-flash", "gemini-3.8-flash", "gemini-1.5-flash"]
    last_err = ""

    for model_id in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                raw_text = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()

                # Clean markdown characters like asterisks (**bold**)
                clean_text = raw_text.replace("**", "").replace("##", "")

                p1_match = re.search(r"Period 1(?:\s+Meaning)?[:\-]\s*(.+)", clean_text, re.IGNORECASE)
                p2_match = re.search(r"Period 2(?:\s+Meaning)?[:\-]\s*(.+)", clean_text, re.IGNORECASE)
                evo_match = re.search(r"(?:Semantic\s+)?Evolution(?:\s+Narrative)?[:\-]\s*(.+)", clean_text, re.IGNORECASE | re.DOTALL)

                p1_def = p1_match.group(1).strip().split("\n")[0] if p1_match else ""
                p2_def = p2_match.group(1).strip().split("\n")[0] if p2_match else ""
                evo_def = evo_match.group(1).strip() if evo_match else ""

                # Fallback to lines if regex pattern is bypassed
                if not p1_def or not p2_def:
                    lines = [l.strip() for l in clean_text.split("\n") if l.strip()]
                    if len(lines) >= 2:
                        p1_def = lines[0]
                        p2_def = lines[1]
                        evo_def = " ".join(lines[2:]) if len(lines) > 2 else f"Wasserstein shift: {distance:.4f}."

                if p1_def and p2_def:
                    return p1_def, p2_def, evo_def, ""

        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read().decode('utf-8')[:200]}"
            if e.code == 404:
                continue
            break
        except Exception as e:
            last_err = str(e)
            break

    # If all models failed
    p1 = f"Anchors: [{', '.join(s1_terms)}]."
    p2 = f"Anchors: [{', '.join(s2_terms)}]."
    evo = f"Wasserstein shift: {distance:.4f}."
    return p1, p2, evo, last_err


# -------------------------------------------------------------
# 5. STREAMLIT INTERFACE
# -------------------------------------------------------------
tokenizer, model, device, weight_status = load_pipeline()

# Resolve Gemini API Key (Streamlit Secrets > Environment > Sidebar Input)
secret_key = ""
try:
    secret_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    pass

resolved_key = secret_key or os.environ.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.subheader("⚙️ Settings")
    st.caption(weight_status)
    custom_key = st.text_input(
        "Gemini API Key",
        value=resolved_key,
        type="password",
        help="Optional: Automatically detected from Cloud secrets if configured."
    )
    final_api_key = custom_key.strip()

st.title("🔬 Semantic Shift Detection")
st.markdown(
    "Distributional semantic divergence detection powered by fine-tuned BERT representations "
    "(Layers 9–12) and Wasserstein Optimal Transport."
)

word_input = st.text_input("Target Word", value="apple", placeholder="e.g., apple, cap, record, cell, tree")

col_p1, col_p2 = st.columns(2)
with col_p1:
    old_sentences = st.text_area(
        "Historical / Canonical Era (Period 1)",
        height=140,
        value="She picked a crisp red apple directly from the garden branch.\n"
              "He baked a warm cinnamon pie filled with sliced green apple.\n"
              "The basket was brimming with fresh cider and sweet ripe apple.",
    )

with col_p2:
    new_sentences = st.text_area(
        "Modern / Shifted Era (Period 2)",
        height=140,
        value="The company released a new operating system update for every apple smartphone.\n"
              "He bought shares of apple stock before the Silicon Valley product launch.\n"
              "Her laptop was serviced at the downtown apple store yesterday.",
    )

if st.button("Analyze Shift", type="primary", use_container_width=True):
    old_list = [s.strip() for s in old_sentences.split("\n") if s.strip()]
    new_list = [s.strip() for s in new_sentences.split("\n") if s.strip()]

    if not old_list or not new_list:
        st.error("Please enter at least one sentence for both periods.")
    else:
        with st.spinner("Computing contextual embeddings & Wasserstein transport..."):
            with torch.no_grad():
                emb1 = get_sentence_embeddings(model, tokenizer, old_list, word_input, device)
                emb2 = get_sentence_embeddings(model, tokenizer, new_list, word_input, device)
                distance = compute_distributional_distance(emb1, emb2)

            if distance < 0.35:
                verdict_status = "STABLE"
                margin = abs(0.35 - distance)
            elif distance < 0.55:
                verdict_status = "DOMAIN"
                margin = min(abs(distance - 0.35), abs(0.55 - distance))
            else:
                verdict_status = "RADICAL"
                margin = abs(distance - 0.55)

            confidence = 2.0 * (1.0 / (1.0 + torch.exp(-torch.tensor(16.0 * margin))) - 0.5).item()

            era1_terms = extract_salient_terms(old_list, word_input)
            era2_terms = extract_salient_terms(new_list, word_input)

            p1_meaning, p2_meaning, evo_summary, api_error = synthesize_meanings(
                word_input, old_list, new_list, era1_terms, era2_terms, distance, final_api_key
            )

        st.subheader("📊 Quantitative Diagnostics")
        v_col, m_col, c_col = st.columns([1.5, 1, 1])

        with v_col:
            if verdict_status == "STABLE":
                st.success("### ✅ STABLE / NO SHIFT")
            elif verdict_status == "DOMAIN":
                st.warning("### ⚠️ DOMAIN / EMERGING SHIFT")
            else:
                st.error("### 🚨 RADICAL SEMANTIC SHIFT")

        with m_col:
            st.metric("Wasserstein Metric", f"{distance:.4f}")
        with c_col:
            st.metric("Confidence Interval", f"{confidence * 100:.1f}%")

        st.subheader("📖 Semantic Meaning & Sense Evolution")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.info(f"**Period 1 Meaning (Canonical)**\n\n{p1_meaning}")
        with col_m2:
            st.info(f"**Period 2 Meaning (Modern)**\n\n{p2_meaning}")

        if evo_summary:
            st.markdown(f"**Evolution Narrative:** {evo_summary}")

        if api_error:
            st.warning(f"⚠️ Gemini API Notice: {api_error}")

        st.subheader("🏷️ Salient Context Anchors")
        st.markdown(f"**Era 1 Anchors:** {format_chips_html(era1_terms, '#2563eb')}", unsafe_allow_html=True)
        st.markdown(f"**Era 2 Anchors:** {format_chips_html(era2_terms, '#7c3aed')}", unsafe_allow_html=True)
# -------------------------------------------------------------
class SemanticShiftEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    def forward(self, encodings, target_indices):
        outputs = self.encoder(**encodings)
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
# 2. CACHED MODEL LOADER (Crucial for Streamlit Cloud Memory)
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Loading calibrated BERT model into cloud memory...")
def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = SemanticShiftEncoder("bert-base-uncased").to(device)

    weights_path = os.path.join(BASE_DIR, "data", "semeval2020_eng", "semantic_encoder_full.pt")
    if os.path.exists(weights_path) and os.path.getsize(weights_path) > 1024:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        status = "Calibrated weights loaded (SemEval-2020 benchmark)."
    else:
        status = "Running on base BERT encoder."

    model.eval()
    return tokenizer, model, device, status


# -------------------------------------------------------------
# 3. DISTRIBUTIONAL MATH & ANCHORS
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
# 4. SENSE SYNTHESIZER
# -------------------------------------------------------------
def synthesize_meanings(
    word: str,
    s1_list: List[str],
    s2_list: List[str],
    s1_terms: List[str],
    s2_terms: List[str],
    distance: float,
    api_key: str
) -> Tuple[str, str, str]:
    if not api_key:
        p1 = f"Context Anchors: [{', '.join(s1_terms)}]."
        p2 = f"Context Anchors: [{', '.join(s2_terms)}]."
        evo = f"Wasserstein shift distance: {distance:.4f}. (Add a Gemini API key for natural language definitions)."
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

    headers = {"Content-Type": "application/json"}
    payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
    candidate_models = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash-latest"]

    for model_id in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                text = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()

                p1_match = re.search(r"Period 1 Meaning:\s*(.+)", text, re.IGNORECASE)
                p2_match = re.search(r"Period 2 Meaning:\s*(.+)", text, re.IGNORECASE)
                evo_match = re.search(r"Semantic Evolution:\s*(.+)", text, re.IGNORECASE | re.DOTALL)

                p1_def = p1_match.group(1).strip().split("\n")[0] if p1_match else ""
                p2_def = p2_match.group(1).strip().split("\n")[0] if p2_match else ""
                evo_def = evo_match.group(1).strip() if evo_match else ""

                if p1_def and p2_def:
                    return p1_def, p2_def, evo_def
        except Exception:
            continue

    p1 = f"Anchors: [{', '.join(s1_terms)}]."
    p2 = f"Anchors: [{', '.join(s2_terms)}]."
    evo = f"Wasserstein shift: {distance:.4f}."
    return p1, p2, evo


# -------------------------------------------------------------
# 5. STREAMLIT INTERFACE
# -------------------------------------------------------------
tokenizer, model, device, weight_status = load_pipeline()

# Resolve Gemini API Key (Streamlit Secrets > Environment > Sidebar Input)
secret_key = ""
try:
    secret_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    pass

resolved_key = secret_key or os.environ.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.subheader("⚙️ Settings")
    st.caption(weight_status)
    custom_key = st.text_input(
        "Gemini API Key",
        value=resolved_key,
        type="password",
        help="Optional: Automatically detected from Cloud secrets if configured."
    )
    final_api_key = custom_key.strip()

st.title("🔬 Semantic Shift Detection")
st.markdown(
    "Distributional semantic divergence detection powered by fine-tuned BERT representations "
    "(Layers 9–12) and Wasserstein Optimal Transport."
)

word_input = st.text_input("Target Word", value="apple", placeholder="e.g., apple, cap, record, cell, tree")

col_p1, col_p2 = st.columns(2)
with col_p1:
    old_sentences = st.text_area(
        "Historical / Canonical Era (Period 1)",
        height=140,
        value="She picked a crisp red apple directly from the garden branch.\n"
              "He baked a warm cinnamon pie filled with sliced green apple.\n"
              "The basket was brimming with fresh cider and sweet ripe apple.",
    )

with col_p2:
    new_sentences = st.text_area(
        "Modern / Shifted Era (Period 2)",
        height=140,
        value="The company released a new operating system update for every apple smartphone.\n"
              "He bought shares of apple stock before the Silicon Valley product launch.\n"
              "Her laptop was serviced at the downtown apple store yesterday.",
    )

if st.button("Analyze Shift", type="primary", use_container_width=True):
    old_list = [s.strip() for s in old_sentences.split("\n") if s.strip()]
    new_list = [s.strip() for s in new_sentences.split("\n") if s.strip()]

    if not old_list or not new_list:
        st.error("Please enter at least one sentence for both periods.")
    else:
        with st.spinner("Computing contextual embeddings & Wasserstein transport..."):
            with torch.no_grad():
                emb1 = get_sentence_embeddings(model, tokenizer, old_list, word_input, device)
                emb2 = get_sentence_embeddings(model, tokenizer, new_list, word_input, device)
                distance = compute_distributional_distance(emb1, emb2)

            if distance < 0.35:
                verdict_status = "STABLE"
                margin = abs(0.35 - distance)
            elif distance < 0.55:
                verdict_status = "DOMAIN"
                margin = min(abs(distance - 0.35), abs(0.55 - distance))
            else:
                verdict_status = "RADICAL"
                margin = abs(distance - 0.55)

            confidence = 2.0 * (1.0 / (1.0 + torch.exp(-torch.tensor(16.0 * margin))) - 0.5).item()

            era1_terms = extract_salient_terms(old_list, word_input)
            era2_terms = extract_salient_terms(new_list, word_input)

            p1_meaning, p2_meaning, evo_summary = synthesize_meanings(
                word_input, old_list, new_list, era1_terms, era2_terms, distance, final_api_key
            )

        st.subheader("📊 Quantitative Diagnostics")
        v_col, m_col, c_col = st.columns([1.5, 1, 1])

        with v_col:
            if verdict_status == "STABLE":
                st.success("### ✅ STABLE / NO SHIFT")
            elif verdict_status == "DOMAIN":
                st.warning("### ⚠️ DOMAIN / EMERGING SHIFT")
            else:
                st.error("### 🚨 RADICAL SEMANTIC SHIFT")

        with m_col:
            st.metric("Wasserstein Metric", f"{distance:.4f}")
        with c_col:
            st.metric("Confidence Interval", f"{confidence * 100:.1f}%")

        st.subheader("📖 Semantic Meaning & Sense Evolution")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.info(f"**Period 1 Meaning (Canonical)**\n\n{p1_meaning}")
        with col_m2:
            st.info(f"**Period 2 Meaning (Modern)**\n\n{p2_meaning}")

        if evo_summary:
            st.markdown(f"**Evolution Narrative:** {evo_summary}")

        st.subheader("🏷️ Salient Context Anchors")
        st.markdown(f"**Era 1 Anchors:** {format_chips_html(era1_terms, '#2563eb')}", unsafe_allow_html=True)
        st.markdown(f"**Era 2 Anchors:** {format_chips_html(era2_terms, '#7c3aed')}", unsafe_allow_html=True)
