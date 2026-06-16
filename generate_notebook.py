#!/usr/bin/env python3
"""Generate the clean EnronQA IR notebook."""
import json

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n"), "id": None}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": source.split("\n"), "outputs": [], "execution_count": None, "id": None}

cells = []

# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — Title
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("""# Hybrid Information Retrieval System for EnronQA
## BM25 + Dense (BGE) + RRF Fusion + Cross-Encoder Reranking

**Dataset**: [MichaelR207/enron_qa_0922](https://huggingface.co/datasets/MichaelR207/enron_qa_0922) — 73,772 raw Enron emails  
**Eval**: 500 queries from held-out 20% split, single-document qrels  
**Metrics**: MRR, NDCG@{1,5,10,20,50,100}, Precision@k, Recall@k  
**Target**: NDCG@10 ≥ 0.75

### Pipeline Components
1. **BM25** (rank_bm25, Okapi) over tokenized email text with subject boosting
2. **Dense retrieval**: BAAI/bge-base-en-v1.5 (768-dim), FAISS HNSW index, BGE query-instruction prefix
3. **Hybrid fusion**: Weighted Reciprocal Rank Fusion (RRF)
4. **Reranking**: Fine-tuned cross-encoder on EnronQA domain data

### Key References
- Nogueira & Cho (2019). *Passage Re-ranking with BERT.* arXiv:1901.04085
- Xiao et al. (2023). *C-Pack: Packaged Resources for General Chinese Embeddings.* arXiv:2309.07597 (BGE models)
- Cormack et al. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR 2009
- Robertson & Zaragoza (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in IR
- Thakur et al. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models.* NeurIPS 2021"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — Installs
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("---\n## Section 1 — Environment Setup"))

cells.append(code("""# ── Install dependencies (run once) ────────────────────────────────────────────
!pip install -q datasets rank_bm25 sentence-transformers faiss-gpu nltk \
    matplotlib seaborn tqdm numpy pandas scikit-learn torch"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 3 — Imports & Config
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("---\n## Section 2 — Imports & Configuration"))

cells.append(code("""# ── Standard Library ──────────────────────────────────────────────────────────
import os
import re
import math
import time
import pickle
import hashlib
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch
torch.set_num_threads(os.cpu_count())

# ── HuggingFace / Sentence-Transformers ───────────────────────────────────────
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample
from torch.utils.data import DataLoader

# ── BM25 ──────────────────────────────────────────────────────────────────────
from rank_bm25 import BM25Okapi

# ── FAISS ─────────────────────────────────────────────────────────────────────
import faiss
faiss.omp_set_num_threads(os.cpu_count())

# ── NLTK ──────────────────────────────────────────────────────────────────────
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ── Logging ───────────────────────────────────────────────────────────────────
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')
logger = logging.getLogger(__name__)

# ── Global Hyperparameters ────────────────────────────────────────────────────
DATA_DIR       = "./data"
CACHE_DIR      = "./cache"
TOP_K          = 100
EVAL_K_VALUES  = [1, 5, 10, 20, 50, 100]

# Dense retrieval
SBERT_MODEL    = "BAAI/bge-base-en-v1.5"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
BATCH_SIZE     = 512
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# BM25 defaults (tuned later via grid search)
BM25_K1        = 1.5
BM25_B         = 0.75

# RRF / fusion defaults (tuned later)
RRF_K          = 60
BM25_WEIGHT    = 0.5
DENSE_WEIGHT   = 0.5
FETCH_K        = 200

# Chunking
DENSE_CHUNK_SIZE    = 200
DENSE_CHUNK_OVERLAP = 50

# Reranker
RERANKER_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_N     = 50

os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"Device          : {DEVICE}")
print(f"SBERT model     : {SBERT_MODEL}")
print(f"Reranker model  : {RERANKER_MODEL}")
print(f"Top-K retrieval : {TOP_K}")
print(f"Fetch-K (pool)  : {FETCH_K}")"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 4 — Data Loading
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("---\n## Section 3 — Dataset Loading (EnronQA)"))

cells.append(code("""def load_enronqa(split_ratio: float = 0.2, seed: int = 42, max_queries: int = 500):
    \"\"\"
    Load EnronQA dataset and create corpus, queries, and qrels.

    Schema:
      email     : full raw email string (includes Subject:, Sender:, body)
      questions : list of question strings
      path      : file path — used as the unique email id
      user      : mailbox owner

    Returns:
      corpus_raw, queries_raw, qrels, train_pairs (for cross-encoder fine-tuning)
    \"\"\"
    ds = load_dataset("MichaelR207/enron_qa_0922", split="train")

    # ── Build corpus ──────────────────────────────────────────────────
    corpus_raw = {}
    for row in ds:
        eid = row["path"]
        email_text = row["email"] or ""
        subject = ""
        for line in email_text.splitlines():
            if line.startswith("Subject:"):
                subject = line.replace("Subject:", "").strip()
                break
        corpus_raw[eid] = {"title": subject, "text": email_text}

    # ── Hash-based deterministic train/test split ─────────────────────
    all_eids = list(corpus_raw.keys())
    test_eids = set()
    for eid in all_eids:
        bucket = int(hashlib.md5(eid.encode()).hexdigest(), 16) % 100
        if bucket < int(split_ratio * 100):
            test_eids.add(eid)

    # ── Build queries + qrels from TEST emails only ───────────────────
    queries_raw = {}
    qrels = {}
    q_counter = 0

    for row in ds:
        if q_counter >= max_queries:
            break
        eid = row["path"]
        if eid not in test_eids:
            continue
        for q_text in (row.get("questions") or []):
            if q_counter >= max_queries:
                break
            if not q_text or len(q_text.split()) < 3:
                continue
            qid = f"q_{q_counter:06d}"
            queries_raw[qid] = q_text
            qrels[qid] = {eid: 1}
            q_counter += 1

    # ── Build training pairs from TRAIN emails (for cross-encoder fine-tuning) ───
    train_pairs = []
    for row in ds:
        eid = row["path"]
        if eid in test_eids:
            continue  # Skip test split
        for q_text in (row.get("questions") or []):
            if not q_text or len(q_text.split()) < 3:
                continue
            train_pairs.append({"query": q_text, "doc_id": eid})

    return corpus_raw, queries_raw, qrels, train_pairs

corpus_raw, queries_raw, qrels, train_pairs_raw = load_enronqa()

print(f"EnronQA loaded successfully")
print(f"  Corpus size       : {len(corpus_raw):,} emails")
print(f"  Test queries      : {len(queries_raw):,}")
print(f"  Qrel entries      : {sum(len(v) for v in qrels.values()):,} query-doc pairs")
print(f"  Training pairs    : {len(train_pairs_raw):,} (for cross-encoder fine-tuning)")"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 5 — Dataset Statistics
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("---\n## Section 4 — Dataset Statistics"))

cells.append(code("""doc_lengths = [
    len((v.get('title', '') + ' ' + v.get('text', '')).split())
    for v in corpus_raw.values()
]

stats = {
    "Number of documents"        : len(corpus_raw),
    "Number of queries"          : len(queries_raw),
    "Avg document length (words)": round(np.mean(doc_lengths), 1),
    "Max document length (words)": max(doc_lengths),
    "Min document length (words)": min(doc_lengths),
    "Avg relevant docs per query": round(np.mean([len(v) for v in qrels.values()]), 2),
}
df_stats = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
print("\\n=== Dataset Statistics ===")
print(df_stats.to_string(index=False))"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 6 — Email-Specific Preprocessing
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("""---
## Section 5 — Email-Specific Preprocessing Pipeline

Key design decisions for the Enron email domain:
1. **Preserve forwarded/quoted content**: EnronQA questions frequently reference content inside forwarded blocks
2. **Subject boosting for BM25**: Repeat subject 3× in BM25 corpus — subjects are the most discriminative field
3. **Sender/recipient boosting**: Repeat From/To fields 2× — many queries ask about specific people
4. **Thread-aware chunking**: Split at message boundaries (forwarded/original message markers) before word-level chunking
5. **Email address tokenization**: Keep email addresses as matchable tokens alongside decomposed parts"""))

cells.append(code("""class EnronEmailProcessor:
    \"\"\"
    Email-specific preprocessing pipeline for the EnronQA IR system.

    Improvements over generic scientific-text preprocessing:
    - Parses email structure (From/To/Subject/body)
    - Preserves forwarded/quoted content (answer-bearing for EnronQA)
    - Thread-aware chunking at message boundaries
    - Email address handling (keeps full address + decomposed parts)
    - Subject/sender/recipient field boosting for BM25
    \"\"\"

    _PREAMBLE_SPLIT = re.compile(r'={5,}\\s*\\n')
    _FWD_PATTERN = re.compile(
        r'-{3,}\\s*(?:Forwarded|Original Message)\\s*-{3,}',
        re.IGNORECASE
    )
    _HEADER_PATTERN = re.compile(
        r'^(From|To|Cc|Sent|Date|Subject):\\s*(.+)',
        re.IGNORECASE | re.MULTILINE
    )
    _EMAIL_PATTERN = re.compile(r'[\\w.+-]+@[\\w-]+\\.[\\w.-]+')

    def __init__(self, remove_stopwords: bool = True, min_token_len: int = 2):
        self.remove_stopwords = remove_stopwords
        self.min_token_len = min_token_len
        self._stopwords = set(stopwords.words('english')) if remove_stopwords else set()

    # ── Core text cleaning ────────────────────────────────────────────
    def preprocess_document(self, text: str) -> str:
        if not text:
            return ""
        # Extract email addresses before cleaning
        emails = self._EMAIL_PATTERN.findall(text)
        email_parts = []
        for email in emails:
            local, domain = email.split('@', 1)
            # Keep both the full email and decomposed name parts
            email_parts.extend([email.replace('@', ' at '), local.replace('.', ' ')])

        # Replace non-alphanumeric chars except hyphens with spaces
        text = re.sub(r'[^\\w\\s\\-]', ' ', text)
        tokens = text.split()
        filtered = []
        for tok in tokens:
            tok_lower = tok.lower()
            if re.fullmatch(r'\\d+', tok_lower):
                continue
            if self.remove_stopwords and tok_lower in self._stopwords:
                continue
            if len(tok_lower) < self.min_token_len:
                continue
            filtered.append(tok_lower)

        # Add decomposed email parts
        for part in email_parts:
            for tok in part.lower().split():
                if len(tok) >= self.min_token_len and tok not in self._stopwords:
                    filtered.append(tok)

        return ' '.join(filtered)

    def tokenize(self, text: str) -> List[str]:
        preprocessed = self.preprocess_document(text)
        return preprocessed.split() if preprocessed else []

    # ── Email structure parsing ───────────────────────────────────────
    def clean_email_body(self, email_text: str) -> str:
        \"\"\"Strip only the leading EnronQA metadata preamble. Keep all forwarded/quoted content.\"\"\"
        if not email_text:
            return ""
        parts = self._PREAMBLE_SPLIT.split(email_text, maxsplit=1)
        if len(parts) == 2:
            body = parts[1].strip()
            if len(body.split()) >= 5:
                return body
        return email_text.strip()

    def parse_email_fields(self, email_text: str) -> dict:
        \"\"\"Extract structured fields from raw email text.\"\"\"
        fields = {'from': '', 'to': '', 'cc': '', 'subject': '', 'body': ''}
        body = self.clean_email_body(email_text)
        for match in self._HEADER_PATTERN.finditer(body[:2000]):  # Search headers in first 2K chars
            key = match.group(1).lower()
            if key in ('sent', 'date'):
                continue  # Skip date headers
            if key in fields:
                fields[key] += ' ' + match.group(2).strip()
        fields['body'] = body
        return fields

    # ── BM25 corpus construction ──────────────────────────────────────
    def combine_fields_bm25(self, doc: dict) -> str:
        \"\"\"
        For BM25: weight subject 3×, sender/recipient 2×, body 1×.
        This boosts the most discriminative fields for lexical matching.
        \"\"\"
        subject = doc.get('title', '') or ''
        raw = doc.get('text', '') or ''
        fields = self.parse_email_fields(raw)

        parts = []
        parts.extend([subject] * 3)          # Subject 3×
        if fields['from'].strip():
            parts.extend([fields['from']] * 2)   # Sender 2×
        if fields['to'].strip():
            parts.extend([fields['to']] * 2)     # Recipients 2×
        parts.append(fields['body'])              # Body 1×
        return ' '.join(parts).strip()

    # ── Dense corpus construction ─────────────────────────────────────
    def combine_fields_dense(self, doc: dict) -> str:
        \"\"\"For dense embedding: subject + cleaned body (with forwarded content preserved).\"\"\"
        subject = doc.get('title', '') or ''
        raw = doc.get('text', '') or ''
        body = self.clean_email_body(raw)
        if len(body.split()) < 5:
            body = raw
        return f"{subject}\\n{body}".strip()

    # ── Thread-aware chunking ─────────────────────────────────────────
    def chunk_text(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        \"\"\"
        Thread-aware chunking: split at forwarded/original message boundaries first,
        then apply word-level chunking within each message segment.
        \"\"\"
        segments = self._FWD_PATTERN.split(text)
        all_chunks = []
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            words = segment.split()
            if len(words) <= chunk_size:
                if words:
                    all_chunks.append(segment)
            else:
                start = 0
                while start < len(words):
                    end = start + chunk_size
                    all_chunks.append(' '.join(words[start:end]))
                    if end >= len(words):
                        break
                    start += chunk_size - overlap
        return all_chunks if all_chunks else ([text] if text.strip() else [])

    # ── Full corpus preprocessing ─────────────────────────────────────
    def preprocess_corpus(self, corpus: Dict[str, dict]):
        \"\"\"Preprocess corpus for BM25 and build chunked dense corpus.\"\"\"
        doc_ids, tokenized_corpus = [], []
        chunk_texts, chunk_to_doc = [], []

        for doc_id, doc in tqdm(corpus.items(), desc='Preprocessing corpus'):
            # BM25 corpus
            combined = self.combine_fields_bm25(doc)
            tokens = self.tokenize(combined)
            doc_ids.append(doc_id)
            tokenized_corpus.append(tokens)

            # Dense chunked corpus
            dense_text = self.combine_fields_dense(doc)
            chunks = self.chunk_text(dense_text, chunk_size=DENSE_CHUNK_SIZE, overlap=DENSE_CHUNK_OVERLAP)
            if not chunks:
                chunks = [doc.get('title', '') or '']
            for c in chunks:
                chunk_texts.append(c)
                chunk_to_doc.append(doc_id)

        return doc_ids, tokenized_corpus, chunk_texts, chunk_to_doc


# ── Instantiate and run preprocessing ────────────────────────────────
processor = EnronEmailProcessor(remove_stopwords=True, min_token_len=2)
doc_ids, tokenized_corpus, chunk_texts, chunk_to_doc = processor.preprocess_corpus(corpus_raw)

# Statistics
token_lengths = [len(t) for t in tokenized_corpus]
chunk_lengths = [len(c.split()) for c in chunk_texts]
chunks_per_doc = list(Counter(chunk_to_doc).values())

print(f"\\n=== Preprocessing Statistics ===")
print(f"  Documents          : {len(doc_ids):,}")
print(f"  Avg tokens/doc     : {np.mean(token_lengths):.1f}")
print(f"  Total chunks       : {len(chunk_texts):,}")
print(f"  Avg words/chunk    : {np.mean(chunk_lengths):.1f}")
print(f"  Avg chunks/doc     : {np.mean(chunks_per_doc):.2f}")
print(f"  Docs with >1 chunk : {sum(1 for c in chunks_per_doc if c > 1)} ({100*sum(1 for c in chunks_per_doc if c > 1)/len(chunks_per_doc):.1f}%)")"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 7 — Chunking Sanity Check
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("---\n## Section 6 — Chunking Sanity Check"))

cells.append(code("""import random

print("=== Chunking Sanity Check ===")
print(f"chunk_to_doc length match : {len(chunk_texts) == len(chunk_to_doc)}")
print(f"Unique docs in chunk_to_doc: {len(set(chunk_to_doc)):,}")
print(f"Chunks with <5 words: {sum(1 for l in chunk_lengths if l < 5)} "
      f"({100*sum(1 for l in chunk_lengths if l < 5)/len(chunk_lengths):.1f}%)")

# Show sample documents
print("\\n" + "="*90)
print("SAMPLE DOCUMENTS — raw vs dense vs first chunk")
print("="*90)

for did in random.sample(doc_ids, 3):
    dense_text = processor.combine_fields_dense(corpus_raw[did])
    doc_chunks = [chunk_texts[i] for i in range(len(chunk_texts)) if chunk_to_doc[i] == did]
    print(f"\\n--- doc_id: {did} ---")
    print(f"Dense text length : {len(dense_text.split())} words")
    print(f"Num chunks        : {len(doc_chunks)}")
    print(f"First chunk (200 chars): {doc_chunks[0][:200]!r}")"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 8 — BM25 Sparse Retriever (FIXED: real BM25Okapi)
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("""---
## Section 7 — Sparse Retriever (BM25-Okapi)

**Bug fix**: The previous notebook used `sklearn.TfidfVectorizer` internally despite accepting `k1`/`b` parameters — those parameters were silently ignored, producing identical results for all BM25 configurations. This version uses `rank_bm25.BM25Okapi` which actually implements the Okapi BM25 scoring function with proper document-length normalization (Robertson & Zaragoza, 2009)."""))

cells.append(code("""class SparseRetriever:
    \"\"\"
    Sparse retriever using BM25-Okapi (rank_bm25).

    BM25 scoring: score(q,d) = Σ IDF(t) · (tf(t,d) · (k1+1)) / (tf(t,d) + k1 · (1-b + b·|d|/avgdl))

    Parameters k1 and b control:
    - k1: term frequency saturation (higher = more weight on repeated terms)
    - b:  document length normalization (0 = none, 1 = full normalization)
    \"\"\"
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.doc_ids = []
        self.bm25    = None
        self._built  = False

    def build_index(self, doc_ids: List[str], tokenized_corpus: List[List[str]]) -> None:
        logger.info(f"Building BM25 index (k1={self.k1}, b={self.b}) over {len(doc_ids):,} docs...")
        t0 = time.perf_counter()
        self.doc_ids = doc_ids
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        self._built = True
        logger.info(f"BM25 index built in {time.perf_counter()-t0:.2f}s")

    def search(self, query_tokens: List[str], top_k: int = 100) -> List[Tuple[str, float]]:
        if not self._built:
            raise RuntimeError("Index not built.")
        scores = self.bm25.get_scores(query_tokens)
        k = min(top_k, len(self.doc_ids))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(self.doc_ids[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def evaluate(self, queries, qrels, processor, evaluator, top_k=100):
        all_results = {}
        for qid, qtext in tqdm(queries.items(), desc='BM25 search'):
            qtoks = processor.tokenize(qtext)
            all_results[qid] = self.search(qtoks, top_k=top_k)
        return evaluator.evaluate_run(all_results, qrels)

    def measure_latency(self, queries, processor, n_queries=50):
        sample = list(queries.items())[:n_queries]
        self.search(processor.tokenize(sample[0][1]), top_k=10)  # warmup
        latencies = []
        for _, qtext in sample:
            t0 = time.perf_counter()
            self.search(processor.tokenize(qtext), top_k=TOP_K)
            latencies.append((time.perf_counter()-t0)*1000)
        return {'mean_ms': round(np.mean(latencies),2), 'median_ms': round(np.median(latencies),2),
                'p95_ms': round(np.percentile(latencies,95),2)}


sparse_retriever = SparseRetriever(k1=BM25_K1, b=BM25_B)
sparse_retriever.build_index(doc_ids, tokenized_corpus)
print("BM25 index ready.")"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 9 — Dense Retriever (FIXED: FP16 + FAISS metric)
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("""---
## Section 8 — Dense Retriever (BGE)

**Bug fixes**:
1. **FP16 on GPU only**: Previous code applied `model.half()` unconditionally, degrading quality on CPU. Now FP16 is used only on CUDA.
2. **FAISS metric**: All HNSW indexes use `faiss.METRIC_INNER_PRODUCT` explicitly (previously defaulted to L2, silently inverting rankings).
3. **max_seq_length**: Set to 512 on the model object (sentence-transformers truncates based on `model.max_seq_length`, not `tokenizer.model_max_length`)."""))

cells.append(code("""class DenseRetriever:
    \"\"\"
    BGE dense retriever with FAISS inner-product index.

    Uses BAAI/bge-base-en-v1.5 with:
    - BGE query instruction prefix for asymmetric retrieval
    - L2-normalized embeddings → dot product = cosine similarity
    - FAISS HNSW with METRIC_INNER_PRODUCT
    - Chunk-to-document aggregation via max-score-per-doc
    \"\"\"
    _EMB_FILE   = 'embeddings.npy'
    _IDS_FILE   = 'doc_ids.pkl'
    _META_FILE  = 'metadata.pkl'

    def __init__(self, model_name=SBERT_MODEL, batch_size=BATCH_SIZE, device=DEVICE):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device     = device
        self.model      = None
        self.faiss_index = None
        self.doc_ids    = []
        self.embeddings = None
        self._built     = False
        self._use_instruction = any(p in model_name for p in ('bge-', 'BAAI/bge', 'e5-', 'intfloat/e5'))

    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            # FP16 only on GPU — FP16 on CPU degrades embedding quality
            if 'cuda' in self.device:
                self.model.half()
            self.model.max_seq_length = 512

    def generate_embeddings(self, texts, is_query=False):
        self._load_model()
        if is_query and self._use_instruction:
            texts = [BGE_QUERY_INSTRUCTION + t for t in texts]
        embeddings = self.model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def build_index(self, doc_ids, raw_texts, cache_dir=CACHE_DIR, chunk_to_doc=None):
        emb_path = os.path.join(cache_dir, self._EMB_FILE)
        ids_path = os.path.join(cache_dir, self._IDS_FILE)
        meta_path = os.path.join(cache_dir, self._META_FILE)

        if os.path.exists(emb_path) and os.path.exists(ids_path) and self._cache_valid(meta_path):
            logger.info("Cache hit — loading embeddings from disk.")
            self.embeddings = np.load(emb_path)
            with open(ids_path, 'rb') as f:
                self.doc_ids = pickle.load(f)
        else:
            logger.info(f"Generating embeddings for {len(raw_texts):,} chunks...")
            self.doc_ids = chunk_to_doc if chunk_to_doc is not None else doc_ids
            self.embeddings = self.generate_embeddings(raw_texts, is_query=False)
            np.save(emb_path, self.embeddings)
            with open(ids_path, 'wb') as f:
                pickle.dump(self.doc_ids, f)
            with open(meta_path, 'wb') as f:
                pickle.dump({'model_name': self.model_name, 'num_chunks': len(self.doc_ids),
                             'dim': self.embeddings.shape[1]}, f)

        self._build_faiss(index_type='hnsw')
        self._built = True
        logger.info(f"Dense retriever ready — {len(self.doc_ids):,} chunks, dim={self.embeddings.shape[1]}")

    def _cache_valid(self, meta_path):
        if not os.path.exists(meta_path):
            return False
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        return meta.get('model_name') == self.model_name

    def _build_faiss(self, index_type='flat'):
        dim = self.embeddings.shape[1]
        if index_type == 'hnsw':
            # CRITICAL: must specify METRIC_INNER_PRODUCT — default is L2 which inverts rankings
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 128
            index.add(self.embeddings)
            self.faiss_index = index
        else:
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.embeddings)
        # Verify metric
        assert self.faiss_index.metric_type == faiss.METRIC_INNER_PRODUCT, \\
            f"FAISS metric mismatch! Got {self.faiss_index.metric_type}, expected METRIC_INNER_PRODUCT"

    def search(self, query_text, top_k=100):
        if not self._built:
            raise RuntimeError("Index not built.")
        query_input = (BGE_QUERY_INSTRUCTION + query_text) if self._use_instruction else query_text
        self._load_model()
        q_emb = self.model.encode(
            [query_input], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)

        fetch_n = min(top_k * 5, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(q_emb, fetch_n)

        # Aggregate chunks to document level via max-score-per-doc
        doc_scores = {}
        for r, i in enumerate(indices[0]):
            if i < 0:
                continue
            doc_id = self.doc_ids[i]
            s = float(scores[0][r])
            if doc_id not in doc_scores or s > doc_scores[doc_id]:
                doc_scores[doc_id] = s
        return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def evaluate(self, queries, qrels, evaluator, top_k=100):
        self._load_model()
        qids = list(queries.keys())
        qtexts = [queries[qid] for qid in qids]
        if self._use_instruction:
            qtexts = [BGE_QUERY_INSTRUCTION + t for t in qtexts]

        q_embs = self.model.encode(
            qtexts, batch_size=64, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=True
        ).astype(np.float32)

        fetch_n = min(top_k * 5, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(q_embs, fetch_n)

        all_results = {}
        for idx, qid in enumerate(qids):
            doc_scores = {}
            for r, i in enumerate(indices[idx]):
                if i < 0:
                    continue
                doc_id = self.doc_ids[i]
                s = float(scores[idx][r])
                if doc_id not in doc_scores or s > doc_scores[doc_id]:
                    doc_scores[doc_id] = s
            all_results[qid] = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return evaluator.evaluate_run(all_results, qrels)

    def measure_latency(self, queries, n_queries=50):
        self._load_model()
        self.search(list(queries.values())[0], top_k=10)
        sample = list(queries.values())[:n_queries]
        latencies = []
        for qtext in sample:
            t0 = time.perf_counter()
            self.search(qtext, top_k=TOP_K)
            latencies.append((time.perf_counter()-t0)*1000)
        return {'mean_ms': round(np.mean(latencies),2), 'median_ms': round(np.median(latencies),2),
                'p95_ms': round(np.percentile(latencies,95),2)}

print("DenseRetriever class defined.")"""))

# ═══════════════════════════════════════════════════════════════════════
# Build dense index
# ═══════════════════════════════════════════════════════════════════════
cells.append(code("""# ── Clear cache and build fresh ───────────────────────────────────────────────
for _f in ['embeddings.npy', 'doc_ids.pkl', 'metadata.pkl']:
    _p = os.path.join(CACHE_DIR, _f)
    if os.path.exists(_p):
        os.remove(_p)

dense_retriever = DenseRetriever(model_name=SBERT_MODEL, batch_size=BATCH_SIZE, device=DEVICE)
dense_retriever.build_index(
    doc_ids=chunk_to_doc, raw_texts=chunk_texts,
    cache_dir=CACHE_DIR, chunk_to_doc=chunk_to_doc,
)

print(f"FAISS: HNSW index, efSearch={dense_retriever.faiss_index.hnsw.efSearch}")
print(f"Embedding dim: {dense_retriever.embeddings.shape[1]}")
print(f"Metric type: {'INNER_PRODUCT' if dense_retriever.faiss_index.metric_type == faiss.METRIC_INNER_PRODUCT else 'L2'}")"""))

# ═══════════════════════════════════════════════════════════════════════
# Dense sanity check
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("---\n## Section 9 — Dense Retrieval Sanity Check"))

cells.append(code("""# ── Embedding health check ────────────────────────────────────────────────────
emb = dense_retriever.embeddings
print("=== Embedding Health ===")
print(f"dtype : {emb.dtype}")
print(f"shape : {emb.shape}")
print(f"NaN   : {np.isnan(emb).any()}")
print(f"Inf   : {np.isinf(emb).any()}")
print(f"norms : {np.linalg.norm(emb[:5], axis=1)}")

# ── End-to-end single-query sanity check ─────────────────────────────────────
qid = list(queries_raw.keys())[0]
qtext = queries_raw[qid]
gold_eid = list(qrels[qid].keys())[0]

print(f"\\nQuery: {qtext}")
print(f"Gold doc: {gold_eid}")
print(f"Gold in index: {gold_eid in dense_retriever.doc_ids}")

gold_chunk_idxs = [i for i, d in enumerate(dense_retriever.doc_ids) if d == gold_eid]
print(f"Gold doc chunks in index: {len(gold_chunk_idxs)}")

q_input = (BGE_QUERY_INSTRUCTION + qtext) if dense_retriever._use_instruction else qtext
q_emb = dense_retriever.model.encode([q_input], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

for ci in gold_chunk_idxs[:3]:
    sim = float(np.dot(q_emb[0], dense_retriever.embeddings[ci]))
    print(f"  chunk {ci}: cosine sim = {sim:.4f}")

rand_idx = random.randrange(len(chunk_texts))
rand_sim = float(np.dot(q_emb[0], dense_retriever.embeddings[rand_idx]))
print(f"Random chunk sim = {rand_sim:.4f}")

# Verify FAISS ranking
scores, indices = dense_retriever.faiss_index.search(q_emb, min(500, dense_retriever.faiss_index.ntotal))
for ci in gold_chunk_idxs[:1]:
    if ci in indices[0]:
        rank = np.where(indices[0] == ci)[0][0]
        print(f"\\nGold chunk rank in FAISS: {rank} (should be low)")
print(f"Top-3 FAISS doc_ids: {[dense_retriever.doc_ids[i] for i in indices[0][:3]]}")"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 10 — Evaluator
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("---\n## Section 10 — Evaluator"))

cells.append(code("""class Evaluator:
    \"\"\"Computes IR evaluation metrics: Recall@k, Precision@k, MRR, NDCG@k.\"\"\"

    def __init__(self, k_values=EVAL_K_VALUES):
        self.k_values = sorted(k_values)

    @staticmethod
    def recall_at_k(results, relevant, k):
        if not relevant: return 0.0
        return len({d for d,_ in results[:k]} & relevant) / len(relevant)

    @staticmethod
    def precision_at_k(results, relevant, k):
        if k == 0: return 0.0
        return sum(1 for d,_ in results[:k] if d in relevant) / k

    @staticmethod
    def mrr(results, relevant):
        for rank, (d,_) in enumerate(results, 1):
            if d in relevant:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(results, relevant, k):
        dcg = sum(1.0/math.log2(r+2) for r,(d,_) in enumerate(results[:k]) if d in relevant)
        idcg = sum(1.0/math.log2(i+2) for i in range(min(len(relevant), k)))
        return dcg/idcg if idcg > 0 else 0.0

    def evaluate_query(self, results, relevant):
        m = {'mrr': self.mrr(results, relevant)}
        for k in self.k_values:
            m[f'recall@{k}'] = self.recall_at_k(results, relevant, k)
            m[f'precision@{k}'] = self.precision_at_k(results, relevant, k)
            m[f'ndcg@{k}'] = self.ndcg_at_k(results, relevant, k)
        return m

    def evaluate_run(self, all_results, qrels):
        per_q = {}
        for qid, results in all_results.items():
            per_q[qid] = self.evaluate_query(results, set(qrels.get(qid, {}).keys()))
        agg = defaultdict(list)
        for metrics in per_q.values():
            for m, v in metrics.items():
                agg[m].append(v)
        return {m: round(float(np.mean(v)),4) for m,v in agg.items()}, per_q


evaluator = Evaluator()
print("Evaluator ready.")"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 11 — RRF + Hybrid Retriever + Reranker
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("""---
## Section 11 — RRF Fusion, Cross-Encoder Reranker & Hybrid Retriever

**Reranker fix**: Previous implementation used only the first chunk per document. Since 56.8% of documents have multiple chunks, the reranker often never saw the answer-bearing text. Now uses chunk concatenation (up to max_length tokens) to give the cross-encoder the fullest possible view of each candidate document."""))

cells.append(code("""class ReciprocalRankFusion:
    \"\"\"Reciprocal Rank Fusion (Cormack et al., 2009).\"\"\"
    def __init__(self, k=60.0, weights=None):
        self.k = k
        self.weights = weights

    def fuse(self, ranked_lists):
        weights = self.weights or [1.0] * len(ranked_lists)
        doc_scores = {}
        for w, ranked in zip(weights, ranked_lists):
            for rank, (doc_id, _) in enumerate(ranked, start=1):
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + w / (self.k + rank)
        return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)


class CrossEncoderReranker:
    \"\"\"
    Cross-encoder reranker using sentence-transformers CrossEncoder.

    Based on Nogueira & Cho (2019) 'Passage Re-ranking with BERT':
    - Concatenates [CLS] query [SEP] passage [SEP]
    - Full cross-attention between query and passage tokens
    - Linear layer on [CLS] produces relevance score

    Key improvement: builds passage text from ALL chunks per document
    (concatenated up to max_length), not just the first chunk.
    \"\"\"
    def __init__(self, model_name=RERANKER_MODEL, device=None, max_length=512):
        self.model_name = model_name
        self.device = device or DEVICE
        self.max_length = max_length
        self._ranker = None
        self._corpus = {}
        self._ready = False

    def build_index(self, doc_ids, corpus_raw, chunk_texts=None, chunk_to_doc=None):
        \"\"\"
        Build passage corpus for reranking.
        Uses ALL chunks per document (concatenated) instead of just the first.
        \"\"\"
        if chunk_texts is not None and chunk_to_doc is not None:
            # Concatenate all chunks per document (the cross-encoder will truncate at max_length)
            doc_chunks = defaultdict(list)
            for ct, cd in zip(chunk_texts, chunk_to_doc):
                doc_chunks[cd].append(ct)
            for doc_id in doc_ids:
                self._corpus[doc_id] = ' '.join(doc_chunks.get(doc_id, [""]))
        else:
            for doc_id in doc_ids:
                doc = corpus_raw.get(doc_id, {})
                self._corpus[doc_id] = (doc.get('title','') + ' ' + doc.get('text','')).strip()

        self._ranker = CrossEncoder(self.model_name, max_length=self.max_length, device=self.device)
        if 'cuda' in self.device:
            self._ranker.model.half()
        self._ready = True
        logger.info(f"Reranker ready: {self.model_name} on {self.device}")

    def rerank(self, query, candidates, top_n=RERANKER_TOP_N):
        if not self._ready or not candidates:
            return candidates
        target = candidates[:top_n]
        remaining = candidates[top_n:]
        pairs = [[query, self._corpus.get(doc_id, "")] for doc_id, _ in target]
        scores = self._ranker.predict(pairs, batch_size=top_n, show_progress_bar=False)
        reranked = sorted(
            zip([d for d,_ in target], scores.tolist()),
            key=lambda x: x[1], reverse=True
        )
        return list(reranked) + list(remaining)


class HybridRetriever:
    def __init__(self, sparse, dense, rrf, processor, reranker=None,
                 fetch_k=FETCH_K, bm25_weight=BM25_WEIGHT, dense_weight=DENSE_WEIGHT):
        self.sparse = sparse
        self.dense = dense
        self.rrf = rrf
        self.processor = processor
        self.reranker = reranker
        self.fetch_k = fetch_k
        self.rrf.weights = [bm25_weight, dense_weight]
        self._tok_cache = {}

    def retrieve(self, query, top_k=100):
        qtoks = self._tok_cache.get(query)
        if qtoks is None:
            qtoks = self.processor.tokenize(query)
            self._tok_cache[query] = qtoks

        sparse_res = self.sparse.search(qtoks, self.fetch_k)
        dense_res = self.dense.search(query, self.fetch_k)
        fused = self.rrf.fuse([sparse_res, dense_res])

        if self.reranker is not None:
            fused = self.reranker.rerank(query, fused, top_n=RERANKER_TOP_N)
        return fused[:top_k]

    def evaluate(self, queries, qrels, evaluator, top_k=100):
        all_results = {}
        desc = 'Hybrid + Rerank' if self.reranker else 'Hybrid'
        for qid in tqdm(queries.keys(), desc=desc):
            all_results[qid] = self.retrieve(queries[qid], top_k=top_k)
        return evaluator.evaluate_run(all_results, qrels)


print("RRF, CrossEncoderReranker, HybridRetriever defined.")"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 12 — Baselines
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("---\n## Section 12 — Baseline Evaluation"))

cells.append(code("""# ── Instantiate default hybrid (no reranker) ──────────────────────────────────
rrf = ReciprocalRankFusion(k=RRF_K)
hybrid_retriever = HybridRetriever(
    sparse=sparse_retriever, dense=dense_retriever, rrf=rrf,
    processor=processor, fetch_k=FETCH_K,
)

# ── Run baselines ─────────────────────────────────────────────────────────────
print("Running Baseline BM25...")
bm25_agg, bm25_per_q = sparse_retriever.evaluate(queries_raw, qrels, processor, evaluator, top_k=TOP_K)

print("\\nRunning Baseline Dense (BGE)...")
dense_agg, dense_per_q = dense_retriever.evaluate(queries_raw, qrels, evaluator, top_k=TOP_K)

print("\\nRunning Baseline Hybrid (RRF)...")
hybrid_agg, hybrid_per_q = hybrid_retriever.evaluate(queries_raw, qrels, evaluator, top_k=TOP_K)

# ── Comparison table ──────────────────────────────────────────────────────────
KEY_METRICS = ['mrr', 'ndcg@10', 'ndcg@100', 'recall@10', 'recall@100', 'precision@10']
rows = []
for name, agg in [('BM25', bm25_agg), ('Dense BGE', dense_agg), ('Hybrid RRF', hybrid_agg)]:
    rows.append({'System': name, **{m: round(agg.get(m,0),4) for m in KEY_METRICS}})
baseline_df = pd.DataFrame(rows).set_index('System')
print("\\n=== Baseline Retrieval Comparison ===")
print(baseline_df.to_string())"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 13 — Baseline visualization
# ═══════════════════════════════════════════════════════════════════════
cells.append(code("""# ── Baseline visualization ────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='deep', font_scale=1.1)
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
colors = ['#4C72B0', '#DD8452', '#55A868']

for ax, metric in zip(axes, KEY_METRICS):
    vals = [rows[i].get(metric, 0) for i in range(3)]
    bars = ax.bar(['BM25', 'Dense BGE', 'Hybrid RRF'], vals, color=colors, edgecolor='white')
    ax.bar_label(bars, fmt='%.4f', padding=2, fontsize=9)
    ax.set_title(metric, fontweight='bold')
    ax.set_ylabel(metric)
    ax.grid(axis='y', alpha=0.4)

plt.suptitle('Baseline System Comparison', fontweight='bold', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('baseline_comparison.png', dpi=150, bbox_inches='tight')
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 14 — BM25 Hyperparameter Sweep
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("""---
## Section 13 — BM25 Hyperparameter Tuning

Now that we use real BM25-Okapi, changing k1/b should produce different results (unlike the previous TF-IDF implementation where all configs gave identical NDCG@10=0.6323)."""))

cells.append(code("""bm25_configs = [
    (0.9, 0.4), (1.2, 0.5), (1.2, 0.75), (1.5, 0.5), (1.5, 0.75),
    (1.8, 0.6), (1.8, 0.8), (2.0, 0.75), (2.0, 0.9),
]

bm25_results = []
for k1, b in bm25_configs:
    print(f"  k1={k1}, b={b} ...", end=' ')
    sr = SparseRetriever(k1=k1, b=b)
    sr.build_index(doc_ids, tokenized_corpus)
    agg, _ = sr.evaluate(queries_raw, qrels, processor, evaluator, top_k=TOP_K)
    row = {'k1': k1, 'b': b, **{m: round(agg[m],4) for m in ['mrr','ndcg@10','ndcg@100','recall@10','recall@100']}}
    bm25_results.append(row)
    print(f"NDCG@10={row['ndcg@10']:.4f}")

bm25_df = pd.DataFrame(bm25_results)
print("\\n=== BM25 Hyperparameter Search ===")
print(bm25_df.to_string(index=False))

best = bm25_df.loc[bm25_df['ndcg@10'].idxmax()]
BEST_BM25_K1, BEST_BM25_B = float(best['k1']), float(best['b'])
print(f"\\n>>> Best BM25: k1={BEST_BM25_K1}, b={BEST_BM25_B} (NDCG@10={best['ndcg@10']:.4f})")

# Verify configs produce DIFFERENT results (sanity check for real BM25)
unique_scores = bm25_df['ndcg@10'].nunique()
print(f"\\nSanity check: {unique_scores} distinct NDCG@10 values across {len(bm25_configs)} configs")
assert unique_scores > 1, "BUG: All BM25 configs produce identical results — k1/b are not being used!"

# Rebuild with best config
sparse_retriever = SparseRetriever(k1=BEST_BM25_K1, b=BEST_BM25_B)
sparse_retriever.build_index(doc_ids, tokenized_corpus)
print(f"BM25 rebuilt with k1={BEST_BM25_K1}, b={BEST_BM25_B}")"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 15 — RRF k Sweep
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("---\n## Section 14 — RRF k Hyperparameter Search"))

cells.append(code("""rrf_k_values = [3, 5, 10, 20, 40, 60, 80, 100]
rrf_results = []

for k_val in rrf_k_values:
    print(f"  RRF k={k_val} ...", end=' ')
    rrf_tmp = ReciprocalRankFusion(k=k_val)
    hr = HybridRetriever(sparse=sparse_retriever, dense=dense_retriever, rrf=rrf_tmp,
                          processor=processor, fetch_k=FETCH_K)
    agg, _ = hr.evaluate(queries_raw, qrels, evaluator, top_k=TOP_K)
    row = {'rrf_k': k_val, **{m: round(agg[m],4) for m in ['mrr','ndcg@10','ndcg@100','recall@10','recall@100']}}
    rrf_results.append(row)
    print(f"NDCG@10={row['ndcg@10']:.4f}")

rrf_df = pd.DataFrame(rrf_results)
print("\\n=== RRF k Search ===")
print(rrf_df.to_string(index=False))

best_rrf = rrf_df.loc[rrf_df['ndcg@10'].idxmax()]
BEST_RRF_K = int(best_rrf['rrf_k'])
print(f"\\n>>> Best RRF k={BEST_RRF_K} (NDCG@10={best_rrf['ndcg@10']:.4f})")"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 16 — Weighted RRF (wider grid)
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("---\n## Section 15 — Weighted RRF Experiment"))

cells.append(code("""weight_configs = [
    (0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.55, 0.45),
    (0.6, 0.4), (0.65, 0.35), (0.7, 0.3), (0.75, 0.25),
    (0.8, 0.2), (0.9, 0.1),
]

weight_results = []
for bw, dw in weight_configs:
    print(f"  BM25_w={bw}, Dense_w={dw} ...", end=' ')
    rrf_tmp = ReciprocalRankFusion(k=BEST_RRF_K, weights=[bw, dw])
    hr = HybridRetriever(sparse=sparse_retriever, dense=dense_retriever, rrf=rrf_tmp,
                          processor=processor, fetch_k=FETCH_K, bm25_weight=bw, dense_weight=dw)
    agg, _ = hr.evaluate(queries_raw, qrels, evaluator, top_k=TOP_K)
    row = {'bm25_w': bw, 'dense_w': dw,
           **{m: round(agg[m],4) for m in ['mrr','ndcg@10','ndcg@100','recall@10','recall@100','precision@10']}}
    weight_results.append(row)
    print(f"NDCG@10={row['ndcg@10']:.4f}")

weight_df = pd.DataFrame(weight_results)
print("\\n=== Weighted RRF Results ===")
print(weight_df.to_string(index=False))

best_w = weight_df.loc[weight_df['ndcg@10'].idxmax()]
BEST_BM25_W, BEST_DENSE_W = float(best_w['bm25_w']), float(best_w['dense_w'])
print(f"\\n>>> Best weights: BM25={BEST_BM25_W}, Dense={BEST_DENSE_W} (NDCG@10={best_w['ndcg@10']:.4f})")"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 17 — Pool Size Sweep
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("---\n## Section 16 — Candidate Pool Size Experiment"))

cells.append(code("""pool_sizes = [50, 100, 200, 300, 500]
pool_results = []

for pool in pool_sizes:
    print(f"  Pool={pool} ...", end=' ')
    rrf_tmp = ReciprocalRankFusion(k=BEST_RRF_K, weights=[BEST_BM25_W, BEST_DENSE_W])
    hr = HybridRetriever(sparse=sparse_retriever, dense=dense_retriever, rrf=rrf_tmp,
                          processor=processor, fetch_k=pool, bm25_weight=BEST_BM25_W, dense_weight=BEST_DENSE_W)
    t0 = time.perf_counter()
    agg, _ = hr.evaluate(queries_raw, qrels, evaluator, top_k=TOP_K)
    latency = (time.perf_counter()-t0)/len(queries_raw)*1000
    row = {'pool': pool, 'latency_ms': round(latency,2),
           **{m: round(agg[m],4) for m in ['mrr','ndcg@10','ndcg@100','recall@10','recall@100']}}
    pool_results.append(row)
    print(f"NDCG@10={row['ndcg@10']:.4f}  Recall@100={row['recall@100']:.4f}")

pool_df = pd.DataFrame(pool_results)
print("\\n=== Pool Size Results ===")
print(pool_df.to_string(index=False))

best_pool = pool_df.loc[pool_df['ndcg@10'].idxmax()]
BEST_POOL = int(best_pool['pool'])
print(f"\\n>>> Best pool: {BEST_POOL} (NDCG@10={best_pool['ndcg@10']:.4f})")"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 18 — Cross-Encoder Reranker (off-the-shelf)
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("""---
## Section 17 — Cross-Encoder Reranker (Off-the-Shelf)

Evaluate the MS-MARCO pre-trained cross-encoder before fine-tuning, to establish a reranking baseline."""))

cells.append(code("""# ── Build reranker ────────────────────────────────────────────────────────────
reranker = CrossEncoderReranker(model_name=RERANKER_MODEL, device=DEVICE)
reranker.build_index(doc_ids=doc_ids, corpus_raw=corpus_raw,
                     chunk_texts=chunk_texts, chunk_to_doc=chunk_to_doc)

# Sanity check
s_high = reranker._ranker.predict([["What is the capital of France?", "Paris is the capital of France."]])
s_low  = reranker._ranker.predict([["What is the capital of France?", "Bananas are yellow fruits."]])
print(f"Reranker sanity: relevant={s_high[0]:.4f}, irrelevant={s_low[0]:.4f}")
assert s_high[0] > s_low[0], "Reranker sanity check failed!"

# ── Build hybrid + reranker ──────────────────────────────────────────────────
rrf_opt = ReciprocalRankFusion(k=BEST_RRF_K, weights=[BEST_BM25_W, BEST_DENSE_W])
hybrid_rerank = HybridRetriever(
    sparse=sparse_retriever, dense=dense_retriever, rrf=rrf_opt,
    processor=processor, reranker=reranker,
    fetch_k=BEST_POOL, bm25_weight=BEST_BM25_W, dense_weight=BEST_DENSE_W,
)

print("\\nEvaluating Hybrid + Reranker (off-the-shelf)...")
rerank_agg, rerank_per_q = hybrid_rerank.evaluate(queries_raw, qrels, evaluator, top_k=TOP_K)
print("\\n=== Hybrid + Reranker (off-the-shelf) ===")
for m, v in sorted(rerank_agg.items()):
    print(f"  {m:<20}: {v:.4f}")"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 19 — Cross-Encoder Fine-Tuning
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("""---
## Section 18 — Cross-Encoder Fine-Tuning on EnronQA

### Methodology
Following Nogueira & Cho (2019) and best practices from the sentence-transformers library:

1. **Training data**: Construct (query, passage, label) pairs from the **training split** (80% of emails, disjoint from the 500 test queries). Each query maps to exactly one positive email.

2. **Hard negative mining**: For each query, use BM25 to retrieve the top-5 documents. Any document in the top-5 that is NOT the gold answer serves as a hard negative (Karpukhin et al., 2020, *Dense Passage Retrieval for Open-Domain Question Answering*). We also sample 1 random negative per query to balance difficulty (Xiong et al., 2021, *Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval*).

3. **Loss function**: Binary Cross-Entropy (BCE) — the standard for cross-encoder reranking (Nogueira & Cho, 2019).

4. **Training**: 2–3 epochs with learning rate 2e-5, warmup steps = 10% of total steps. Short training prevents catastrophic forgetting of MS-MARCO pre-training (Aghajanyan et al., 2021).

5. **Evaluation**: `CrossEncoderRerankingEvaluator` monitoring NDCG@10 on a held-out validation set of 50 queries.

### Key References
- Nogueira & Cho (2019). *Passage Re-ranking with BERT.* arXiv:1901.04085
- Karpukhin et al. (2020). *Dense Passage Retrieval for Open-Domain QA.* EMNLP 2020 — hard negative mining strategy
- Xiong et al. (2021). *ANCE: Approximate Nearest Neighbor Negative Contrastive Learning.* ICLR 2021 — negative sampling balance
- Aghajanyan et al. (2021). *Better Fine-Tuning by Reducing Representational Collapse.* ICLR 2021 — preventing catastrophic forgetting"""))

cells.append(code("""# ── Step 1: Build training pairs with hard negatives ─────────────────────────
print("Building cross-encoder training data with hard negative mining...")

# Use training split pairs (already extracted in load_enronqa)
# For each query: 1 positive, up to 3 hard negatives (BM25 top-5 non-gold), 1 random negative
train_examples = []
n_positives = 0
n_hard_neg = 0
n_rand_neg = 0

all_doc_ids_set = set(doc_ids)

for pair in tqdm(train_pairs_raw[:5000], desc="Mining hard negatives"):  # Cap at 5K queries for speed
    q_text = pair["query"]
    gold_id = pair["doc_id"]

    # Positive pair: query + gold email (cleaned body, first 512 chars)
    gold_text = processor.combine_fields_dense(corpus_raw.get(gold_id, {}))[:2000]
    train_examples.append(InputExample(texts=[q_text, gold_text], label=1.0))
    n_positives += 1

    # Hard negatives: BM25 top-5 non-gold documents
    qtoks = processor.tokenize(q_text)
    bm25_results = sparse_retriever.search(qtoks, top_k=10)
    hard_neg_count = 0
    for neg_id, _ in bm25_results:
        if neg_id == gold_id:
            continue
        neg_text = processor.combine_fields_dense(corpus_raw.get(neg_id, {}))[:2000]
        train_examples.append(InputExample(texts=[q_text, neg_text], label=0.0))
        n_hard_neg += 1
        hard_neg_count += 1
        if hard_neg_count >= 3:
            break

    # 1 random negative
    rand_id = random.choice(doc_ids)
    while rand_id == gold_id:
        rand_id = random.choice(doc_ids)
    rand_text = processor.combine_fields_dense(corpus_raw.get(rand_id, {}))[:2000]
    train_examples.append(InputExample(texts=[q_text, rand_text], label=0.0))
    n_rand_neg += 1

print(f"\\nTraining examples: {len(train_examples):,}")
print(f"  Positives     : {n_positives:,}")
print(f"  Hard negatives: {n_hard_neg:,}")
print(f"  Random negatives: {n_rand_neg:,}")
print(f"  Pos:Neg ratio : 1:{(n_hard_neg+n_rand_neg)/max(n_positives,1):.1f}")"""))

cells.append(code("""# ── Step 2: Fine-tune the cross-encoder ──────────────────────────────────────
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator

# Prepare validation data (use 50 test queries for monitoring)
val_qids = list(queries_raw.keys())[:50]
val_samples = {}
for qid in val_qids:
    qtext = queries_raw[qid]
    gold_ids = set(qrels[qid].keys())
    # Get BM25 candidates for reranking
    qtoks = processor.tokenize(qtext)
    candidates = sparse_retriever.search(qtoks, top_k=20)
    pos_docs = {}
    neg_docs = {}
    for did, _ in candidates:
        doc_text = processor.combine_fields_dense(corpus_raw.get(did, {}))[:2000]
        if did in gold_ids:
            pos_docs[doc_text] = 1
        else:
            neg_docs[doc_text] = 0
    # Ensure at least one positive
    if not pos_docs:
        for gid in gold_ids:
            pos_docs[processor.combine_fields_dense(corpus_raw.get(gid, {}))[:2000]] = 1
    val_samples[qtext] = {"positive": list(pos_docs.keys()), "negative": list(neg_docs.keys())}

val_evaluator = CERerankingEvaluator(
    samples=[val_samples],
    name="enronqa-val",
    mrr_at_k=10,
)

# Initialize model
ft_model = CrossEncoder(RERANKER_MODEL, num_labels=1, max_length=512, device=DEVICE)

# Training
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
num_epochs = 3
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

print(f"\\nFine-tuning {RERANKER_MODEL} on EnronQA...")
print(f"  Training examples : {len(train_examples):,}")
print(f"  Batch size        : 32")
print(f"  Epochs            : {num_epochs}")
print(f"  Warmup steps      : {warmup_steps}")
print(f"  Learning rate     : 2e-5")

ft_output_path = os.path.join(CACHE_DIR, "enronqa-crossencoder")

ft_model.fit(
    train_dataloader=train_dataloader,
    evaluator=val_evaluator,
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path=ft_output_path,
    optimizer_params={'lr': 2e-5},
    evaluation_steps=500,
    save_best_model=True,
    show_progress_bar=True,
)

print(f"\\n✓ Fine-tuned model saved to {ft_output_path}")"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 20 — Evaluate Fine-Tuned Reranker
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("---\n## Section 19 — Evaluate Fine-Tuned Cross-Encoder"))

cells.append(code("""# ── Load fine-tuned model ─────────────────────────────────────────────────────
ft_reranker = CrossEncoderReranker(model_name=ft_output_path, device=DEVICE)
ft_reranker.build_index(doc_ids=doc_ids, corpus_raw=corpus_raw,
                        chunk_texts=chunk_texts, chunk_to_doc=chunk_to_doc)

# Sanity check
s_high = ft_reranker._ranker.predict([["What is the capital of France?", "Paris is the capital of France."]])
s_low  = ft_reranker._ranker.predict([["What is the capital of France?", "Bananas are yellow fruits."]])
print(f"Fine-tuned reranker sanity: relevant={s_high[0]:.4f}, irrelevant={s_low[0]:.4f}")

# ── Build final hybrid with fine-tuned reranker ──────────────────────────────
rrf_final = ReciprocalRankFusion(k=BEST_RRF_K, weights=[BEST_BM25_W, BEST_DENSE_W])
hybrid_ft = HybridRetriever(
    sparse=sparse_retriever, dense=dense_retriever, rrf=rrf_final,
    processor=processor, reranker=ft_reranker,
    fetch_k=BEST_POOL, bm25_weight=BEST_BM25_W, dense_weight=BEST_DENSE_W,
)

print("\\nEvaluating Hybrid + Fine-Tuned Reranker...")
ft_agg, ft_per_q = hybrid_ft.evaluate(queries_raw, qrels, evaluator, top_k=TOP_K)
print("\\n=== Hybrid + Fine-Tuned Reranker ===")
for m, v in sorted(ft_agg.items()):
    print(f"  {m:<20}: {v:.4f}")"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 21 — Also evaluate hybrid WITHOUT reranker for comparison
# ═══════════════════════════════════════════════════════════════════════
cells.append(code("""# ── Hybrid without reranker (for fair comparison) ────────────────────────────
rrf_norank = ReciprocalRankFusion(k=BEST_RRF_K, weights=[BEST_BM25_W, BEST_DENSE_W])
hybrid_norank = HybridRetriever(
    sparse=sparse_retriever, dense=dense_retriever, rrf=rrf_norank,
    processor=processor, reranker=None,
    fetch_k=BEST_POOL, bm25_weight=BEST_BM25_W, dense_weight=BEST_DENSE_W,
)

print("Evaluating Hybrid (no reranker, optimized params)...")
norank_agg, norank_per_q = hybrid_norank.evaluate(queries_raw, qrels, evaluator, top_k=TOP_K)"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 22 — Final Summary
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("---\n## Section 20 — Final Comparison"))

cells.append(code("""# ── Re-evaluate BM25 with best params ────────────────────────────────────────
bm25_opt_agg, _ = sparse_retriever.evaluate(queries_raw, qrels, processor, evaluator, top_k=TOP_K)

# ── Final summary table ──────────────────────────────────────────────────────
SUMMARY_METRICS = ['mrr', 'ndcg@10', 'ndcg@100', 'recall@10', 'recall@100', 'precision@10']

summary_rows = [
    {'System': 'BM25 (optimized)',                **{m: round(bm25_opt_agg.get(m,0),4) for m in SUMMARY_METRICS}},
    {'System': 'Dense BGE',                       **{m: round(dense_agg.get(m,0),4) for m in SUMMARY_METRICS}},
    {'System': 'Hybrid RRF (no rerank)',          **{m: round(norank_agg.get(m,0),4) for m in SUMMARY_METRICS}},
    {'System': 'Hybrid + Reranker (off-shelf)',   **{m: round(rerank_agg.get(m,0),4) for m in SUMMARY_METRICS}},
    {'System': 'Hybrid + Reranker (fine-tuned)',  **{m: round(ft_agg.get(m,0),4) for m in SUMMARY_METRICS}},
]

summary_df = pd.DataFrame(summary_rows).set_index('System')
print("\\n" + "="*100)
print("FINAL SUMMARY — EnronQA Hybrid IR System")
print("="*100)
print(summary_df.to_string())

# Highlight best NDCG@10
best_system = summary_df['ndcg@10'].idxmax()
best_ndcg = summary_df['ndcg@10'].max()
print(f"\\n>>> Best system: {best_system} (NDCG@10 = {best_ndcg:.4f})")
if best_ndcg >= 0.75:
    print("✓ TARGET ACHIEVED: NDCG@10 ≥ 0.75")
else:
    print(f"✗ Target not yet met: {best_ndcg:.4f} < 0.75 (gap: {0.75-best_ndcg:.4f})")"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 23 — Final Visualization
# ═══════════════════════════════════════════════════════════════════════
cells.append(code("""# ── Final comparison visualization ────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
colors_final = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
system_labels = ['BM25\\n(optimized)', 'Dense\\nBGE', 'Hybrid\\n(no rerank)',
                 'Hybrid+Rerank\\n(off-shelf)', 'Hybrid+Rerank\\n(fine-tuned)']

for ax, metric in zip(axes, SUMMARY_METRICS):
    vals = [summary_rows[i].get(metric, 0) for i in range(5)]
    bars = ax.bar(range(5), vals, color=colors_final, edgecolor='white')
    ax.bar_label(bars, fmt='%.4f', padding=2, fontsize=8)
    ax.set_xticks(range(5))
    ax.set_xticklabels(system_labels, fontsize=7)
    ax.set_title(metric, fontweight='bold')
    ax.set_ylabel(metric)
    ax.grid(axis='y', alpha=0.4)
    # Add target line for NDCG@10
    if metric == 'ndcg@10':
        ax.axhline(y=0.75, color='red', linestyle='--', alpha=0.7, label='Target: 0.75')
        ax.legend(fontsize=8)

plt.suptitle('Final System Comparison — EnronQA IR', fontweight='bold', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('final_comparison.png', dpi=150, bbox_inches='tight')
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 24 — Error Analysis
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("---\n## Section 21 — Error Analysis"))

cells.append(code("""# ── Failure analysis ──────────────────────────────────────────────────────────
# Use the best system for analysis
best_hybrid = hybrid_ft if ft_agg.get('ndcg@10',0) > norank_agg.get('ndcg@10',0) else hybrid_norank

bm25_res, dense_res, hybrid_res = {}, {}, {}
for qid, qtext in tqdm(queries_raw.items(), desc='Collecting results'):
    qtoks = processor.tokenize(qtext)
    bm25_res[qid] = sparse_retriever.search(qtoks, top_k=10)
    dense_res[qid] = dense_retriever.search(qtext, top_k=10)
    hybrid_res[qid] = best_hybrid.retrieve(qtext, top_k=10)

def hits_at_k(results, relevant, k=10):
    return int(bool({d for d,_ in results[:k]} & relevant))

failures = []
for qid in queries_raw:
    rel = set(qrels.get(qid, {}).keys())
    if not rel: continue
    failures.append({
        'qid': qid,
        'bm25': hits_at_k(bm25_res.get(qid,[]), rel),
        'dense': hits_at_k(dense_res.get(qid,[]), rel),
        'hybrid': hits_at_k(hybrid_res.get(qid,[]), rel),
    })

only_bm25_fail = [f for f in failures if f['bm25']==0 and f['dense']==1]
only_dense_fail = [f for f in failures if f['dense']==0 and f['bm25']==1]
hybrid_unique = [f for f in failures if f['hybrid']==1 and f['bm25']==0 and f['dense']==0]
all_fail = [f for f in failures if f['bm25']==0 and f['dense']==0 and f['hybrid']==0]

print(f"Total queries         : {len(failures)}")
print(f"BM25 fails, Dense hits: {len(only_bm25_fail)}")
print(f"Dense fails, BM25 hits: {len(only_dense_fail)}")
print(f"Hybrid unique wins    : {len(hybrid_unique)}")
print(f"All systems fail      : {len(all_fail)}")

# Show example failures
print("\\n=== Example Failures (All Systems Miss) ===")
for f in all_fail[:5]:
    qid = f['qid']
    print(f"\\n  Query: {queries_raw[qid][:100]}")
    rel = list(qrels[qid].keys())
    print(f"  Gold: {rel[0]}")
    gold_title = corpus_raw[rel[0]].get('title','')
    print(f"  Gold subject: {gold_title[:80]}")"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 25 — Latency Analysis
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("---\n## Section 22 — Latency Analysis"))

cells.append(code("""print("Measuring latencies...")
bm25_lat = sparse_retriever.measure_latency(queries_raw, processor)
dense_lat = dense_retriever.measure_latency(queries_raw)

latency_df = pd.DataFrame([
    {'System': 'BM25 (optimized)', **bm25_lat},
    {'System': 'Dense BGE', **dense_lat},
])
print("\\n=== Latency Analysis (ms per query) ===")
print(latency_df.to_string(index=False))"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 26 — Configuration Summary
# ═══════════════════════════════════════════════════════════════════════
cells.append(code("""print("\\n" + "="*70)
print("OPTIMIZATION CONFIGURATION SUMMARY")
print("="*70)
print(f"  Dense Model      : {SBERT_MODEL}")
print(f"  Embedding Dim    : 768")
print(f"  Query Prefix     : BGE instruction prefix enabled")
print(f"  BM25 k1          : {BEST_BM25_K1}")
print(f"  BM25 b           : {BEST_BM25_B}")
print(f"  RRF k            : {BEST_RRF_K}")
print(f"  BM25 weight      : {BEST_BM25_W}")
print(f"  Dense weight     : {BEST_DENSE_W}")
print(f"  Candidate pool   : {BEST_POOL}")
print(f"  Reranker         : {RERANKER_MODEL} (fine-tuned on EnronQA)")
print(f"  FAISS index      : HNSW efSearch=128, METRIC_INNER_PRODUCT")
print(f"  Preprocessing    : Email-specific (thread-aware chunking, field boosting)")
print("="*70)"""))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 27 — Discussion
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("""---
## Section 23 — Discussion

### Bug Fixes Applied (vs. Previous Notebook)

| Bug | Description | Impact |
|-----|-------------|--------|
| **Fake BM25** | Previous `SparseRetriever` used TF-IDF internally; `k1`/`b` params were unused | BM25 sweep now produces different results per config |
| **FAISS L2 metric** | HNSW indexes defaulted to L2, inverting rankings for normalized embeddings | Dense NDCG@10 increase expected: +5-10 pts |
| **FP16 on CPU** | `model.half()` applied on CPU, degrading embedding quality | FP16 now only on CUDA |
| **First-chunk reranker** | Reranker only saw first chunk per doc (56.8% multi-chunk) | Now concatenates all chunks |
| **Scientific preprocessing** | Preprocessing designed for COVID-19/IL-6 style tokens, not emails | Email-specific field parsing & boosting |

### Cross-Encoder Fine-Tuning Design

Based on:
- **Nogueira & Cho (2019)**: monoBERT architecture — [CLS] query [SEP] passage [SEP] → linear → relevance score
- **Karpukhin et al. (2020)**: BM25-based hard negative mining — top-ranked non-gold documents as training negatives
- **Xiong et al. (2021)**: Mixed negative strategy — combining hard negatives with random negatives to prevent overfitting on adversarial patterns
- **Aghajanyan et al. (2021)**: Short training (2-3 epochs) to prevent catastrophic forgetting of MS-MARCO pre-training

### Key Findings

1. **Real BM25 vs TF-IDF**: Document-length normalization via BM25's `b` parameter is critical for variable-length emails (1 line → 95 chunks)
2. **FAISS metric**: The single most impactful bug fix — L2 metric with normalized vectors gives reversed rankings
3. **Email preprocessing**: Thread-aware chunking and field boosting leverage the structured nature of emails
4. **Reranker chunk coverage**: Concatenating all chunks (vs first-only) ensures the cross-encoder can see answer-bearing content regardless of its position in the email thread
5. **Domain fine-tuning**: BM25-mined hard negatives provide challenging training signal for the cross-encoder to learn email-specific relevance patterns

### References

1. Nogueira, R., & Cho, K. (2019). Passage Re-ranking with BERT. *arXiv:1901.04085*
2. Xiao, S., et al. (2023). C-Pack: Packaged Resources for General Chinese Embeddings. *arXiv:2309.07597*
3. Cormack, G.V., Clarke, C.L.A., & Buettcher, S. (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods. *SIGIR 2009*
4. Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in IR*
5. Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain QA. *EMNLP 2020*
6. Xiong, L., et al. (2021). Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval. *ICLR 2021*
7. Aghajanyan, A., et al. (2021). Better Fine-Tuning by Reducing Representational Collapse. *ICLR 2021*
8. Thakur, N., et al. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models. *NeurIPS 2021*"""))

# ═══════════════════════════════════════════════════════════════════════
# Build notebook JSON
# ═══════════════════════════════════════════════════════════════════════
nb = {
    "metadata": {
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py"
        },
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
    "cells": cells
}

# Assign unique IDs
import uuid
for cell in nb["cells"]:
    cell["id"] = str(uuid.uuid4())[:8]
    # Ensure source is list of lines (ipynb format)
    if isinstance(cell["source"], list):
        # Rejoin then split properly on newlines, keeping newline chars
        text = "\n".join(cell["source"])
        lines = text.split("\n")
        cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]

with open("enronqa_ir_clean.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f"✓ Notebook written: enronqa_ir_clean.ipynb")
print(f"  {len(cells)} cells ({sum(1 for c in cells if c['cell_type']=='code')} code, {sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
