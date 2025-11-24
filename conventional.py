#!/usr/bin/env python3
"""
Keywords Recommendation System with Gradio interface.

Usage:
    pip install -r requirements.txt
    # or
    pip install nltk scikit-learn sentence-transformers gensim numpy gradio

Run once to download NLTK data (the script will attempt to download on first run):
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

Then run:
    python keywords_recommender_gradio.py
Open the local Gradio URL printed in the console.

This script wraps a keywords extractor (TF-IDF, RAKE, Embedding) into a simple Gradio UI.
"""

from collections import Counter, defaultdict
import math
import re
import warnings
import numpy as np
import os
import sys
import traceback

# NLTK: ensure resources early
import nltk

def ensure_nltk_data(packages=None):
    """
    Ensure required NLTK packages are available. Attempts to find locally
    and downloads any missing packages. Prints progress for visibility.
    """
    if packages is None:
        packages = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
    for pkg in packages:
        try:
            # tokenizers (punkt, punkt_tab) live under tokenizers/
            if pkg in ("punkt", "punkt_tab"):
                nltk.data.find(f"tokenizers/{pkg}")
            else:
                nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            try:
                print(f"NLTK resource '{pkg}' not found. Downloading...", file=sys.stderr)
                nltk.download(pkg)
            except Exception as e:
                print(f"Warning: failed to download NLTK package '{pkg}': {e}", file=sys.stderr)

# Run check now to avoid runtime surprises
ensure_nltk_data()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try sentence-transformers or fallback to gensim
USE_SBERT = False
_SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
_sbert_model = None

try:
    from sentence_transformers import SentenceTransformer
    USE_SBERT = True
except Exception:
    USE_SBERT = False

# gensim Word2Vec fallback
try:
    from gensim.models import Word2Vec
except Exception:
    Word2Vec = None

# Try to import gradio
try:
    import gradio as gr
except Exception:
    gr = None

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

def preprocess_text(text, lemmatize=True):
    """
    Tokenize, remove stopwords and non-alphanumeric tokens, and optionally lemmatize.
    """
    tokens = []
    if not text:
        return tokens
    # sent_tokenize relies on punkt and punkt_tab
    for sent in sent_tokenize(text):
        for tok in word_tokenize(sent):
            tok = tok.lower()
            if not TOKEN_RE.fullmatch(tok):
                continue
            if tok in STOPWORDS:
                continue
            if lemmatize:
                tok = lemmatizer.lemmatize(tok)
            tokens.append(tok)
    return tokens

# ---------------- TF-IDF ----------------
def tfidf_keywords(text, corpus=None, top_n=10, ngram_range=(1,2)):
    if corpus is None:
        corpus = [text]
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    idx = corpus.index(text) if text in corpus else len(corpus)-1
    doc_vec = tfidf_matrix[idx].toarray()[0]
    pairs = [(feature_names[i], float(doc_vec[i])) for i in range(len(feature_names)) if doc_vec[i] > 0]
    return sorted(pairs, key=lambda x: x[1], reverse=True)[:top_n]

# ---------------- RAKE ----------------
def rake_extract(text, top_n=10, min_length=1, max_length=3):
    sentences = sent_tokenize(text.lower())
    phrase_list = []
    for sent in sentences:
        words = [w for w in word_tokenize(sent)]
        current = []
        for w in words:
            if TOKEN_RE.fullmatch(w) and w not in STOPWORDS:
                current.append(lemmatizer.lemmatize(w))
            else:
                if current:
                    phrase_list.append(" ".join(current))
                    current = []
        if current:
            phrase_list.append(" ".join(current))
    filtered = [p for p in phrase_list if min_length <= len(p.split()) <= max_length]
    if not filtered:
        return []
    word_freq, word_degree = Counter(), Counter()
    for phrase in filtered:
        words = phrase.split()
        degree = len(words) - 1
        for w in words:
            word_freq[w] += 1
            word_degree[w] += degree
    for w in word_freq:
        word_degree[w] += word_freq[w]
    word_score = {w: (word_degree[w] / word_freq[w]) for w in word_freq}
    phrase_scores = {p: sum(word_score[w] for w in p.split() if w in word_score) for p in filtered}
    return sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

# ---------------- Embedding-based ----------------
def candidate_phrases_from_text(text, min_len=1, max_len=3):
    tokens = preprocess_text(text, lemmatize=True)
    phrases = set()
    for n in range(min_len, max_len+1):
        for i in range(0, len(tokens)-n+1):
            phrases.add(" ".join(tokens[i:i+n]))
    return list(phrases)

def _load_sbert_model():
    global _sbert_model
    if _sbert_model is None:
        try:
            _sbert_model = SentenceTransformer(_SBERT_MODEL_NAME)
        except Exception as e:
            print(f"Failed to load SentenceTransformer model '{_SBERT_MODEL_NAME}': {e}", file=sys.stderr)
            _sbert_model = None
    return _sbert_model

def embedding_keywords(text, corpus=None, top_n=10, method='sbert'):
    candidates = candidate_phrases_from_text(text, min_len=1, max_len=3)
    if not candidates:
        return []
    # SBERT path
    if method == 'sbert' and USE_SBERT:
        model = _load_sbert_model()
        if model is None:
            # fallback to word2vec or tfidf
            method = 'word2vec'
        else:
            try:
                texts = [text] + candidates
                embs = model.encode(texts, show_progress_bar=False)
                doc_emb, cand_embs = embs[0], embs[1:]
            except Exception as e:
                print("Error during SBERT encoding:", e, file=sys.stderr)
                method = 'word2vec'
    # Word2Vec fallback
    if method != 'sbert':
        if Word2Vec is None:
            # final fallback: tfidf proxy
            return tfidf_keywords(text, corpus=corpus, top_n=top_n, ngram_range=(1,3))
        # prepare corpus tokens
        if corpus is None:
            corpus_tokens = [preprocess_text(text)]
        else:
            corpus_tokens = [preprocess_text(d) for d in corpus]
        try:
            w2v = Word2Vec(sentences=corpus_tokens, vector_size=100, min_count=1, epochs=50)
        except Exception as e:
            print("Word2Vec training failed, falling back to TF-IDF. Error:", e, file=sys.stderr)
            return tfidf_keywords(text, corpus=corpus, top_n=top_n, ngram_range=(1,3))
        doc_vecs = [w2v.wv[w] for w in preprocess_text(text) if w in w2v.wv]
        doc_emb = np.mean(doc_vecs, axis=0) if doc_vecs else np.zeros(w2v.vector_size)
        cand_embs = []
        for c in candidates:
            words = [w for w in c.split() if w in w2v.wv]
            if not words:
                cand_embs.append(np.zeros(w2v.vector_size))
            else:
                cand_embs.append(np.mean([w2v.wv[w] for w in words], axis=0))
    # compute cosine similarities
    try:
        sims = cosine_similarity([doc_emb], cand_embs)[0]
    except Exception as e:
        print("Cosine similarity failed:", e, file=sys.stderr)
        return []
    scored = list(zip(candidates, sims.tolist()))
    scored = sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]
    return scored

# ---------------- Combined Interface ----------------
def extract_keywords_full(text, corpus=None, top_n=10, methods=("tfidf","rake","embedding"), embedding_method='sbert'):
    results = {}
    if "tfidf" in methods:
        try:
            results["tfidf"] = tfidf_keywords(text, corpus, top_n)
        except Exception as e:
            results["tfidf"] = f"Error: {e}"
    if "rake" in methods:
        try:
            results["rake"] = rake_extract(text, top_n)
        except Exception as e:
            results["rake"] = f"Error: {e}"
    if "embedding" in methods:
        try:
            results["embedding"] = embedding_keywords(text, corpus, top_n, embedding_method)
        except Exception as e:
            results["embedding"] = f"Error: {e}"
    return results

# ---------------- Gradio app ----------------
def run_interface(text, corpus_text, methods_list, top_n, embedding_method):
    if not text or text.strip() == "":
        return "Please provide input text.", "", ""
    corpus = []
    if corpus_text and corpus_text.strip():
        # split corpus by double-newline or newline
        corpus = [p.strip() for p in re.split(r'\n\n+|\n', corpus_text) if p.strip()]
    methods = tuple(m.strip() for m in methods_list.split(",")) if methods_list else ("tfidf","rake","embedding")
    try:
        res = extract_keywords_full(text, corpus=corpus if corpus else None, top_n=int(top_n), methods=methods, embedding_method=embedding_method)
    except Exception as e:
        tb = traceback.format_exc()
        return f"Extraction failed: {e}\n\n{tb}", "", ""
    # Format outputs nicely
    def fmt(x):
        if isinstance(x, str):
            return x
        if not x:
            return ""
        return "\n".join([f"{i+1}. {k} — {round(v,4)}" for i,(k,v) in enumerate(x)])
    tfidf_out = fmt(res.get("tfidf",""))
    rake_out = fmt(res.get("rake",""))
    emb_out = fmt(res.get("embedding",""))
    return tfidf_out, rake_out, emb_out

def main():
    if gr is None:
        raise RuntimeError("Gradio is not installed. Install with: pip install gradio")
    # warm load SBERT model in background if available to reduce first-call latency
    if USE_SBERT:
        try:
            _ = _load_sbert_model()
        except Exception:
            pass

    with gr.Blocks() as demo:
        gr.Markdown("""# Keywords Recommendation System

Enter text and optional corpus (one document per line). Choose methods and click **Extract**.""")
        with gr.Row():
            text_in = gr.Textbox(label="Input text", lines=8, placeholder="Paste your document here...")
            corpus_in = gr.Textbox(label="Optional corpus (one doc per line)", lines=8, placeholder="Optional additional documents to build corpus...")
        methods_in = gr.Dropdown(choices=["tfidf","rake","embedding","tfidf,rake,embedding"], value="tfidf,rake,embedding", label="Methods (comma-separated)")
        embedding_method = gr.Radio(choices=["sbert","word2vec"], value="sbert" if USE_SBERT else "word2vec", label="Embedding method (sbert if available)")
        top_n = gr.Slider(minimum=1, maximum=50, step=1, value=10, label="Top N keywords")
        output_tfidf = gr.Textbox(label="TF-IDF Keywords", lines=8)
        output_rake = gr.Textbox(label="RAKE Keywords", lines=8)
        output_emb = gr.Textbox(label="Embedding Keywords", lines=8)
        run_btn = gr.Button("Extract")
        run_btn.click(run_interface, inputs=[text_in, corpus_in, methods_in, top_n, embedding_method], outputs=[output_tfidf, output_rake, output_emb])
        gr.Markdown("""**Notes:** If Sentence-BERT isn't installed, the app will try Word2Vec. For best embedding results install `sentence-transformers`.""")
    demo.launch()

if __name__ == "__main__":
    main()
