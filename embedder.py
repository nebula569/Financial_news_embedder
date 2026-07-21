"""
embedder.py
──────────────────────────────────────────────────────────────────────────────
Converts each scraped article into a fixed-length embedding vector and
attaches two types of labels:

  1. SENTIMENT LABEL (self-supervised, no market data needed)
     Uses ProsusAI/finbert — a BERT model fine-tuned on financial text.
     Labels: positive | negative | neutral
     Score:  float in [-1, +1]

  2. MARKET-MOVEMENT LABEL (requires Nifty 50 OHLCV, optional)
     next_day_up : 1 if Nifty 50 closed higher the next trading day, else 0
     next_day_return : actual % change (regression target)

EMBEDDING CHOICES (in order of quality for finance NLP):
  ── FinBERT [CLS] token (768-d)          ← DEFAULT, best for finance text
  ── SBERT finance (384-d)                ← faster, smaller, good for retrieval
  ── Mean-pooled BERT-base (768-d)        ← generic fallback

Each article row in the output DataFrame has:
  • All scraped fields (title, date, section, author, summary, full_text, url)
  • embedding        : numpy array of shape (768,) or (384,)
  • sentiment_label  : "positive" | "negative" | "neutral"
  • sentiment_score  : float ∈ [-1, 1]
  • next_day_up      : int  0/1  (if market data provided)
  • next_day_return  : float     (if market data provided)
──────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, pipeline as hf_pipeline

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

MODELS = {
    # Best for financial sentiment + embeddings
    "finbert": {
        "hf_name":   "ProsusAI/finbert",
        "dim":        768,
        "task":       "text-classification",
        "max_tokens": 512,
    },
    # Smaller / faster sentence-level embeddings (finance domain)
    "finance-sbert": {
        "hf_name":   "nickmuchi/finance-embeddings-investopedia",
        "dim":        384,
        "task":       "feature-extraction",
        "max_tokens": 256,
    },
    # Generic BERT fallback
    "bert-base": {
        "hf_name":   "bert-base-uncased",
        "dim":        768,
        "task":       "feature-extraction",
        "max_tokens": 512,
    },
}

DEFAULT_MODEL = "finbert"
BATCH_SIZE    = 16          # reduce if you hit OOM on GPU
LOG_EVERY     = 50          # log progress every N articles

# ══════════════════════════════════════════════════════════════════════════════
# HELPER: DEVICE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ══════════════════════════════════════════════════════════════════════════════
# FINBERT EMBEDDER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class FinBERTEmbedder:
    """
    Wraps ProsusAI/finbert (or any BERT-family model) to produce:
      • 768-d [CLS] token embedding per article
      • Sentiment label + score from the classification head

    The [CLS] token encodes a global summary of the whole input sequence
    and is the standard choice for document-level classification tasks.
    """

    def __init__(self, model_key: str = DEFAULT_MODEL, device: str = None):
        cfg = MODELS[model_key]
        self.model_key  = model_key
        self.hf_name    = cfg["hf_name"]
        self.dim        = cfg["dim"]
        self.max_tokens = cfg["max_tokens"]
        self.device     = device or best_device()

        logging.info(f"Loading {self.hf_name} on {self.device}…")
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_name)

        # Encoder (produces embeddings)
        self.encoder = AutoModel.from_pretrained(self.hf_name)
        self.encoder.eval()
        self.encoder.to(self.device)

        # Classifier head for sentiment (FinBERT only)
        if model_key == "finbert":
            self.clf = hf_pipeline(
                "text-classification",
                model=self.hf_name,
                tokenizer=self.tokenizer,
                return_all_scores=True,
                truncation=True,
                max_length=self.max_tokens,
                device=0 if self.device == "cuda" else -1,
            )
        else:
            self.clf = None

        logging.info(f"Model ready | dim={self.dim} | device={self.device}")

    # ── embedding ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of texts → numpy array of shape (N, dim).

        Internally processed in mini-batches to avoid OOM.
        Truncates each text to max_tokens tokens.
        """
        all_vecs = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt",
            ).to(self.device)

            out = self.encoder(**enc)

            # [CLS] token is the first token of last_hidden_state
            cls_vec = out.last_hidden_state[:, 0, :]    # (batch, dim)
            all_vecs.append(cls_vec.cpu().float().numpy())

        return np.vstack(all_vecs)   # (N, dim)

    # ── sentiment ─────────────────────────────────────────────────────────────
    def sentiment(self, texts: list[str]) -> list[dict]:
        """
        Returns list of {"label": str, "score": float} for each text.
        score ∈ [-1, +1]  (positive − negative probability)
        """
        if self.clf is None:
            return self._rule_based_sentiment(texts)

        results = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            raw = self.clf(batch)
            for item in raw:
                prob = {r["label"]: r["score"] for r in item}
                label = max(prob, key=prob.get)
                score = round(prob.get("positive", 0) - prob.get("negative", 0), 4)
                results.append({"label": label, "score": score})
        return results

    def _rule_based_sentiment(self, texts: list[str]) -> list[dict]:
        """Simple keyword fallback when no classifier is loaded."""
        POS = {"surge","rally","gain","rise","bullish","profit","growth",
               "outperform","record high","upgrade","beat","strong","robust","boost"}
        NEG = {"fall","drop","decline","loss","bearish","crash","weak","miss",
               "downgrade","plunge","slump","concern","risk","cut","losses"}
        out = []
        for t in texts:
            words = set(t.lower().split())
            p = len(words & POS)
            n = len(words & NEG)
            if p > n:
                out.append({"label": "positive", "score": round((p-n)/(p+n+1e-9), 4)})
            elif n > p:
                out.append({"label": "negative", "score": round((p-n)/(p+n+1e-9), 4)})
            else:
                out.append({"label": "neutral", "score": 0.0})
        return out

    # ── combined encode ────────────────────────────────────────────────────────
    def encode_articles(self, articles: list[dict]) -> pd.DataFrame:
        """
        Full pipeline: article dicts → DataFrame with embeddings + sentiment.

        Input article must have at minimum: title, summary, full_text, date, url
        """
        log = logging.getLogger("embedder")
        n = len(articles)
        log.info(f"Encoding {n} articles with {self.hf_name}…")

        # Build the text to embed for each article.
        # Strategy: title + summary is enough for sentiment classification
        # and gives a reliable document-level signal.
        # We also embed a longer version (title + summary + first-500-chars of body)
        # for the vector representation.
        short_texts = [            # for sentiment
            f"{a.get('title','')}. {a.get('summary','')}"
            for a in articles
        ]
        long_texts = [             # for embedding
            f"{a.get('title','')}. {a.get('summary','')}. "
            f"{a.get('full_text','')[:500]}"
            for a in articles
        ]

        # --- Sentiment --------------------------------------------------
        log.info("  Running sentiment classification…")
        sents = self.sentiment(short_texts)

        # --- Embeddings -------------------------------------------------
        log.info("  Running embedding extraction…")
        vecs = self.embed(long_texts)          # (N, 768)

        # --- Assemble DataFrame -----------------------------------------
        rows = []
        for i, art in enumerate(articles):
            if i % LOG_EVERY == 0:
                log.info(f"  Processed {i}/{n}")
            row = {
                "date":            art.get("date", ""),
                "title":           art.get("title", ""),
                "section":         art.get("section", ""),
                "author":          art.get("author", ""),
                "summary":         art.get("summary", ""),
                "full_text":       art.get("full_text", ""),
                "url":             art.get("url", ""),
                "sentiment_label": sents[i]["label"],
                "sentiment_score": sents[i]["score"],
                "embedding":       vecs[i],        # numpy array (768,)
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

        log.info(f"  Done. DataFrame shape: {df.shape}")
        return df

# ══════════════════════════════════════════════════════════════════════════════
# MARKET-MOVEMENT LABELER
# ══════════════════════════════════════════════════════════════════════════════

def attach_market_labels(
    df: pd.DataFrame,
    lag: int = 1,
) -> pd.DataFrame:
    """
    Download Nifty 50 daily OHLCV and attach two labels to each article:

      next_day_up     : 1 if Nifty 50 close[date + lag] > close[date], else 0
      next_day_return : actual % return of Nifty 50 on date + lag

    lag=1 means "the trading day AFTER the article was published" — this is
    the standard no-look-ahead setup for market prediction models.

    Requires: pip install yfinance
    """
    log = logging.getLogger("embedder")
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed. Skipping market labels. "
                    "Run: pip install yfinance")
        return df

    start = (df["date"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end   = (df["date"].max() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    log.info(f"Downloading Nifty 50 data ({start} → {end})…")
    nifty = yf.download("^NSEI", start=start, end=end,
                        auto_adjust=True, progress=False)
    if nifty.empty:
        log.warning("Nifty 50 download returned empty DataFrame.")
        return df

    nifty = nifty[["Close"]].reset_index()
    nifty.columns = ["date", "nifty_close"]
    nifty["date"] = pd.to_datetime(nifty["date"])
    nifty = nifty.sort_values("date").reset_index(drop=True)

    # Compute next-trading-day return (shift(-lag) gives the future close)
    nifty["nifty_next_close"] = nifty["nifty_close"].shift(-lag)
    nifty["next_day_return"]  = (
        (nifty["nifty_next_close"] - nifty["nifty_close"])
        / nifty["nifty_close"]
    ).round(6)
    nifty["next_day_up"] = (nifty["next_day_return"] > 0).astype(int)

    # Merge on date
    # Articles are matched to the Nifty close on their publication date.
    # The label is the NEXT day's move (lag=1).
    df["_date_only"] = df["date"].dt.normalize()
    nifty["_date_only"] = nifty["date"].dt.normalize()

    df = df.merge(
        nifty[["_date_only", "next_day_return", "next_day_up"]],
        on="_date_only", how="left"
    ).drop(columns="_date_only")

    labeled = df["next_day_up"].notna().sum()
    log.info(f"Market labels attached to {labeled}/{len(df)} articles.")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# SAVE / LOAD EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

def save_embeddings(df: pd.DataFrame, out_dir: str):
    """
    Save two files:
      1. articles_metadata.csv  — all scalar columns (no embedding column)
      2. embeddings.npy         — numpy array of shape (N, dim)

    Keeping them separate lets you load just the metadata into pandas
    and the embeddings into numpy/PyTorch without any CSV parsing overhead.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Separate embedding column from the rest
    emb_matrix = np.vstack(df["embedding"].values)          # (N, dim)
    meta_df    = df.drop(columns=["embedding"])

    meta_path = Path(out_dir) / "articles_metadata.csv"
    emb_path  = Path(out_dir) / "embeddings.npy"

    meta_df.to_csv(meta_path, index=False, encoding="utf-8-sig")
    np.save(str(emb_path), emb_matrix)

    logging.getLogger("embedder").info(
        f"Saved:\n"
        f"  Metadata CSV : {meta_path}  ({len(meta_df)} rows)\n"
        f"  Embeddings   : {emb_path}   shape={emb_matrix.shape}"
    )
    return str(meta_path), str(emb_path)


def load_embeddings(out_dir: str) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Inverse of save_embeddings.
    Returns (metadata_df, embeddings_array).
    """
    meta_path = Path(out_dir) / "articles_metadata.csv"
    emb_path  = Path(out_dir) / "embeddings.npy"

    meta_df    = pd.read_csv(meta_path, encoding="utf-8-sig")
    emb_matrix = np.load(str(emb_path))

    meta_df["date"] = pd.to_datetime(meta_df["date"], errors="coerce")
    return meta_df, emb_matrix


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY-POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_embedder(
    articles: list[dict],
    out_dir: str = "bs_data",
    model_key: str = DEFAULT_MODEL,
    add_market_labels: bool = True,
    market_lag: int = 1,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Complete embedding pipeline.

    Parameters
    ----------
    articles          : list of article dicts from the scraper
    out_dir           : directory for saving outputs
    model_key         : one of "finbert" | "finance-sbert" | "bert-base"
    add_market_labels : download Nifty 50 and attach next-day movement labels
    market_lag        : how many trading days ahead to label (default 1)

    Returns
    -------
    (metadata_df, embeddings_matrix)
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("embedder")

    if not articles:
        raise ValueError("No articles provided to embedder.")

    # 1. Load model and encode
    embedder = FinBERTEmbedder(model_key=model_key)
    df = embedder.encode_articles(articles)

    # 2. Attach market-movement labels (optional)
    if add_market_labels:
        df = attach_market_labels(df, lag=market_lag)

    # 3. Save
    save_embeddings(df, out_dir)

    # Also save full DataFrame with embedding as pickle for convenience
    df.to_pickle(str(Path(out_dir) / "articles_with_embeddings.pkl"))
    log.info(f"Full DataFrame (with embeddings) saved as pickle.")

    # Return metadata + numpy matrix separately
    emb_matrix = np.vstack(df["embedding"].values)
    return df, emb_matrix
