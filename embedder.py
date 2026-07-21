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

MODELS = {
    "finbert": {
        "hf_name":   "ProsusAI/finbert",
        "dim":        768,
        "task":       "text-classification",
        "max_tokens": 512,
    },
    "finance-sbert": {
        "hf_name":   "nickmuchi/finance-embeddings-investopedia",
        "dim":        384,
        "task":       "feature-extraction",
        "max_tokens": 256,
    },
    "bert-base": {
        "hf_name":   "bert-base-uncased",
        "dim":        768,
        "task":       "feature-extraction",
        "max_tokens": 512,
    },
}

DEFAULT_MODEL = "finbert"
BATCH_SIZE    = 16
LOG_EVERY     = 50

def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class FinBERTEmbedder:
    def __init__(self, model_key: str = DEFAULT_MODEL, device: str = None):
        cfg = MODELS[model_key]
        self.model_key  = model_key
        self.hf_name    = cfg["hf_name"]
        self.dim        = cfg["dim"]
        self.max_tokens = cfg["max_tokens"]
        self.device     = device or best_device()

        logging.info(f"Loading {self.hf_name} on {self.device}…")
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_name)

        self.encoder = AutoModel.from_pretrained(self.hf_name)
        self.encoder.eval()
        self.encoder.to(self.device)

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

    @torch.no_grad()
    def embed(self, texts: list[str]) -> np.ndarray:
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

            cls_vec = out.last_hidden_state[:, 0, :]
            all_vecs.append(cls_vec.cpu().float().numpy())

        return np.vstack(all_vecs)

    def sentiment(self, texts: list[str]) -> list[dict]:
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

    def encode_articles(self, articles: list[dict]) -> pd.DataFrame:
        log = logging.getLogger("embedder")
        n = len(articles)
        log.info(f"Encoding {n} articles with {self.hf_name}…")

        short_texts = [
            f"{a.get('title','')}. {a.get('summary','')}"
            for a in articles
        ]
        long_texts = [
            f"{a.get('title','')}. {a.get('summary','')}. "
            f"{a.get('full_text','')[:500]}"
            for a in articles
        ]

        log.info("  Running sentiment classification…")
        sents = self.sentiment(short_texts)

        log.info("  Running embedding extraction…")
        vecs = self.embed(long_texts)

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
                "embedding":       vecs[i],
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

        log.info(f"  Done. DataFrame shape: {df.shape}")
        return df

def attach_market_labels(
    df: pd.DataFrame,
    lag: int = 1,
) -> pd.DataFrame:
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

    nifty["nifty_next_close"] = nifty["nifty_close"].shift(-lag)
    nifty["next_day_return"]  = (
        (nifty["nifty_next_close"] - nifty["nifty_close"])
        / nifty["nifty_close"]
    ).round(6)
    nifty["next_day_up"] = (nifty["next_day_return"] > 0).astype(int)

    df["_date_only"] = df["date"].dt.normalize()
    nifty["_date_only"] = nifty["date"].dt.normalize()

    df = df.merge(
        nifty[["_date_only", "next_day_return", "next_day_up"]],
        on="_date_only", how="left"
    ).drop(columns="_date_only")

    labeled = df["next_day_up"].notna().sum()
    log.info(f"Market labels attached to {labeled}/{len(df)} articles.")
    return df

def save_embeddings(df: pd.DataFrame, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    emb_matrix = np.vstack(df["embedding"].values)
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
    meta_path = Path(out_dir) / "articles_metadata.csv"
    emb_path  = Path(out_dir) / "embeddings.npy"

    meta_df    = pd.read_csv(meta_path, encoding="utf-8-sig")
    emb_matrix = np.load(str(emb_path))

    meta_df["date"] = pd.to_datetime(meta_df["date"], errors="coerce")
    return meta_df, emb_matrix


def run_embedder(
    articles: list[dict],
    out_dir: str = "bs_data",
    model_key: str = DEFAULT_MODEL,
    add_market_labels: bool = True,
    market_lag: int = 1,
) -> tuple[pd.DataFrame, np.ndarray]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("embedder")

    if not articles:
        raise ValueError("No articles provided to embedder.")

    embedder = FinBERTEmbedder(model_key=model_key)
    df = embedder.encode_articles(articles)

    if add_market_labels:
        df = attach_market_labels(df, lag=market_lag)

    save_embeddings(df, out_dir)

    df.to_pickle(str(Path(out_dir) / "articles_with_embeddings.pkl"))
    log.info(f"Full DataFrame (with embeddings) saved as pickle.")

    emb_matrix = np.vstack(df["embedding"].values)
    return df, emb_matrix
