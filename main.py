import argparse
import json
import logging
from pathlib import Path

from todays_paper_scraper import run_scraper, START_DATE, END_DATE
from embedder             import run_embedder, MODELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("main")


def run_pipeline(
    out_dir: str        = "bs_todays_paper_data",
    max_dates: int      = None,
    model_key: str      = "finbert",
    market_labels: bool = True,
    skip_scrape: bool   = False,
    market_lag: int     = 1,
):
    log.info("=" * 65)
    log.info("  Business Standard — Today's Paper Pipeline")
    log.info(f"  Period : {START_DATE.strftime('%Y-%m-%d')} → "
             f"{END_DATE.strftime('%Y-%m-%d')}")
    log.info("  Driver : undetected-chromedriver")
    log.info("  Extract: driver.page_source + BeautifulSoup")
    log.info("=" * 65)

    raw_path = Path(out_dir) / "raw_articles.json"

    # ── Step 1: Scrape ───────────────────────────────────────────────────────
    if skip_scrape and raw_path.exists():
        log.info("[Step 1] Loading existing articles…")
        articles = json.loads(raw_path.read_text("utf-8"))
        log.info(f"  Loaded {len(articles)} articles.")
    else:
        log.info("[Step 1] Scraping Today's Paper…")
        articles = run_scraper(
            out_dir    = out_dir,
            max_dates  = max_dates,
        )

    if not articles:
        log.error("No articles collected. Check scraper.log.")
        return

    log.info(f"  Total articles: {len(articles)}")

    # ── Step 2: Embed + Label ────────────────────────────────────────────────
    log.info(f"\n[Step 2] Embedding {len(articles)} articles with {model_key}…")
    df, emb = run_embedder(
        articles          = articles,
        out_dir           = out_dir,
        model_key         = model_key,
        add_market_labels = market_labels,
        market_lag        = market_lag,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("  PIPELINE COMPLETE")
    log.info("=" * 65)
    log.info(f"  Articles    : {len(df):,}")
    log.info(f"  Date range  : {df['date'].min()} → {df['date'].max()}")
    log.info(f"  Embeddings  : {emb.shape}  (articles × 768)")

    if "sentiment_label" in df.columns:
        log.info("  Sentiment:")
        for label, count in df["sentiment_label"].value_counts().items():
            pct = count / len(df) * 100
            log.info(f"    {label:<12} {count:>6,}  ({pct:.1f}%)")

    if "next_day_up" in df.columns and df["next_day_up"].notna().any():
        up_pct = df["next_day_up"].mean() * 100
        log.info(
            f"  Market labels : {df['next_day_up'].notna().sum():,} articles"
            f" | Up-days: {up_pct:.1f}%"
        )

    log.info(f"\n  Output files in '{out_dir}/':")
    log.info(f"    raw_articles.json            all scraped articles")
    log.info(f"    articles_metadata.csv        scalar columns + labels")
    log.info(f"    embeddings.npy               shape {emb.shape}")
    log.info(f"    articles_with_embeddings.pkl full DataFrame")
    log.info("=" * 65)

    return df, emb


def parse_args():
    p = argparse.ArgumentParser(
        description="BS Today's Paper → FinBERT vectors (10 years)"
    )
    p.add_argument("--out-dir",     default="bs_todays_paper_data",
                   help="Output directory")
    p.add_argument("--test",        action="store_true",
                   help="Scrape 3 dates only (~10 min)")
    p.add_argument("--skip-scrape", action="store_true",
                   help="Skip scraping, embed existing data")
    p.add_argument("--model",       default="finbert",
                   choices=list(MODELS.keys()),
                   help="Embedding model")
    p.add_argument("--no-market",   action="store_true",
                   help="Skip Nifty 50 market labels")
    p.add_argument("--lag",         type=int, default=1,
                   help="Market label lag in trading days")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        out_dir       = args.out_dir,
        max_dates     = 3 if args.test else None,
        model_key     = args.model,
        market_labels = not args.no_market,
        skip_scrape   = args.skip_scrape,
        market_lag    = args.lag,
    )
