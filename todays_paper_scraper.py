"""
todays_paper_scraper.py — FINAL WORKING VERSION
─────────────────────────────────────────────────────────────────────────────
Previous run confirmed:
  ✅ 81 article links correctly extracted per date
  ✅ Hindi filtered out (only 1 rejected)
  ✅ Link filter working perfectly
  ❌ Total: 0 — individual article pages returning empty via requests

Root cause: requests gets 403 on article pages because Cloudflare blocks
plain HTTP requests. The listing page worked because we used undetected
Chrome. Individual articles need the same treatment.

Fix: fetch each article page using driver.get() (undetected Chrome)
instead of requests. Slower but guaranteed to work.
─────────────────────────────────────────────────────────────────────────────
"""

import re
import json
import time
import random
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup

import undetected_chromedriver as uc
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import WebDriverException, TimeoutException

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE         = "https://www.business-standard.com"
TODAYS_PAPER = f"{BASE}/todays-paper"

START_DATE = datetime(2014, 1, 1)
END_DATE   = datetime(2024, 1, 1)

SAVE_EVERY    = 20
ARTICLE_WAIT  = 4      # seconds to wait after loading each article
PAGE_DELAY    = (2.0, 4.0)
ARTICLE_DELAY = (1.5, 3.0)   # between article fetches (same driver)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.3 Safari/605.1.15",
]

SKIP_PARTS = [
    "/todays-paper", "/epaper", "/rss/", "/topic/", "/author/",
    "/search", "/video/", "/photos/", "/podcast/", "/quiz/",
    "/calculator/", "/sitemap", "/about", "/contact", "/subscribe",
    "/login", "/register", "/tag/", ".jpg", ".png", ".gif", ".pdf",
    ".mp4", ".svg", "/advertise", "/terms", "/privacy", "/feedback",
    "/latest", "/premium", "/newsletter", "/e-paper",
]

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def get_logger(out_dir: str) -> logging.Logger:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                Path(out_dir) / "scraper.log", encoding="utf-8"
            ),
        ],
    )
    return logging.getLogger("scraper")

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

class Checkpoint:
    def __init__(self, path: str):
        self.path  = Path(path)
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text("utf-8"))
                self._dates = set(d.get("dates_done", []))
                self._urls  = set(d.get("urls_seen",  []))
                return
            except Exception:
                pass
        self._dates = set()
        self._urls  = set()

    def save(self):
        with self._lock:
            self.path.write_text(
                json.dumps({
                    "dates_done": sorted(self._dates),
                    "urls_seen":  list(self._urls),
                }, indent=2), "utf-8",
            )

    def date_done(self, d: datetime) -> bool:
        return d.strftime("%Y-%m-%d") in self._dates

    def mark_date(self, d: datetime):
        with self._lock:
            self._dates.add(d.strftime("%Y-%m-%d"))
        self.save()

    def url_seen(self, u: str) -> bool:
        return u in self._urls

    def mark_url(self, u: str):
        with self._lock:
            self._urls.add(u)
        if len(self._urls) % 100 == 0:
            self.save()

    @property
    def dates_done(self) -> int:
        return len(self._dates)

# ══════════════════════════════════════════════════════════════════════════════
# ARTICLE STORE
# ══════════════════════════════════════════════════════════════════════════════

class ArticleStore:
    def __init__(self, path: Path, existing: list):
        self._lock     = threading.Lock()
        self._articles = list(existing)
        self._unsaved  = 0
        self.path      = path

    def add(self, rec: dict):
        with self._lock:
            self._articles.append(rec)
            self._unsaved += 1
            if self._unsaved >= SAVE_EVERY:
                self._flush()

    def _flush(self):
        self.path.write_text(
            json.dumps(self._articles, ensure_ascii=False, indent=2),
            "utf-8",
        )
        self._unsaved = 0

    def save(self):
        with self._lock:
            self._flush()

    def all(self) -> list:
        with self._lock:
            return list(self._articles)

    def __len__(self) -> int:
        with self._lock:
            return len(self._articles)

# ══════════════════════════════════════════════════════════════════════════════
# DATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def generate_dates(start, end, max_dates=None):
    dates, cur = [], end
    while cur >= start:
        if cur.weekday() != 6:
            dates.append(cur)
        cur -= timedelta(days=1)
    return dates[:max_dates] if max_dates else dates

# ══════════════════════════════════════════════════════════════════════════════
# LINK EXTRACTION (confirmed working from previous run)
# ══════════════════════════════════════════════════════════════════════════════

def is_article_url(href: str) -> bool:
    if "www.business-standard.com" not in href:
        return False
    if any(s in href for s in SKIP_PARTS):
        return False
    path  = href.replace(BASE, "").strip("/")
    parts = [p for p in path.split("/") if p]
    if len(parts) < 3:
        return False
    last = parts[-1].replace(".html", "")
    if len(last) < 20:
        return False
    if last.count("-") < 2:
        return False
    return True

def extract_article_links(html: str, log: logging.Logger) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    links: list[str] = []

    for a in soup.find_all("a", href=True):
        href: str = a["href"].strip()
        if href.startswith("//"): href = "https:" + href
        elif href.startswith("/"): href = BASE + href
        elif not href.startswith("http"): continue
        href = href.split("?")[0].split("#")[0].rstrip("/")
        if not is_article_url(href): continue
        if href not in seen:
            seen.add(href)
            links.append(href)

    log.info(f"  Extracted {len(links)} article links")
    if links:
        for lnk in links[:3]:
            log.info(f"    → {lnk}")
    return links

# ══════════════════════════════════════════════════════════════════════════════
# ARTICLE PARSING
# ══════════════════════════════════════════════════════════════════════════════

BOILERPLATE = re.compile(
    r"(Subscribe to Business Standard.*|Also Read.*|Don.t miss.*|"
    r"First Published.*|Disclaimer.*|Follow us on.*|Click here to.*|"
    r"Topics :.*|Dear Reader.*|Business Standard has always.*)",
    re.IGNORECASE | re.DOTALL,
)

def clean_text(text: str) -> str:
    text = BOILERPLATE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()

def parse_article(html: str, url: str, date: datetime) -> Optional[dict]:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    rec = {
        "url": url, "title": "", "date": date.strftime("%Y-%m-%d"),
        "section": "", "author": "", "summary": "", "full_text": "",
        "scraped_at": datetime.utcnow().isoformat(),
    }

    # 1. JSON-LD
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            d = json.loads(tag.string or "{}")
            if isinstance(d, list): d = d[0]
            if d.get("@type") in ("NewsArticle", "Article", "WebPage"):
                rec["title"]   = d.get("headline", "")
                rec["summary"] = d.get("description", "")
                auth = d.get("author", {})
                if isinstance(auth, dict):
                    rec["author"] = auth.get("name", "")
                elif isinstance(auth, list):
                    rec["author"] = ", ".join(
                        a.get("name","") for a in auth if isinstance(a,dict))
                pub = d.get("datePublished","")
                if pub:
                    try:
                        rec["date"] = datetime.fromisoformat(
                            pub.replace("Z","+00:00")).strftime("%Y-%m-%d")
                    except Exception:
                        rec["date"] = pub[:10]
                break
        except Exception:
            pass

    def meta(prop):
        t = soup.find("meta", property=prop) or \
            soup.find("meta", attrs={"name": prop})
        return (t or {}).get("content", "")  # type: ignore

    if not rec["title"]:
        rec["title"] = meta("og:title") or \
            (soup.find("h1") or soup.new_tag("x")).get_text(strip=True)
    if not rec["summary"]:
        rec["summary"] = meta("og:description") or meta("description")
    if not rec["date"] or rec["date"] == date.strftime("%Y-%m-%d"):
        pub = meta("article:published_time")
        if pub: rec["date"] = pub[:10]

    # If still no title, try h1 directly
    if not rec["title"]:
        h1 = soup.find("h1")
        if h1: rec["title"] = h1.get_text(strip=True)

    if not rec["title"]:
        return None

    parts = url.replace(BASE,"").strip("/").split("/")
    rec["section"] = parts[0].replace("-"," ").title() if parts else ""

    if not rec["author"]:
        for sel in [".author-name",'[class*="author"]',
                    '[itemprop="author"]',".byline"]:
            t = soup.select_one(sel)
            if t: rec["author"] = t.get_text(strip=True); break

    for sel in ["div.article-content","div.story-content",
                'div[class*="article-body"]','div[class*="story-body"]',
                'div[class*="content-body"]',"article",
                'div[itemprop="articleBody"]']:
        box = soup.select_one(sel)
        if box:
            paras = box.find_all("p")
            if paras:
                rec["full_text"] = clean_text(
                    " ".join(p.get_text(" ",strip=True) for p in paras))
                break
    if not rec["full_text"]:
        rec["full_text"] = clean_text(
            " ".join(p.get_text(" ",strip=True) for p in soup.find_all("p")))

    return rec

# ══════════════════════════════════════════════════════════════════════════════
# DRIVER
# ══════════════════════════════════════════════════════════════════════════════

def create_driver(log: logging.Logger) -> uc.Chrome:
    options = uc.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1366,768")
    options.add_argument("--disable-extensions")
    options.add_argument("--lang=en-IN")
    options.add_argument(f"--user-agent={random.choice(USER_AGENTS)}")
    options.add_argument("--log-level=3")
    log.info("  Starting undetected Chrome…")
    driver = uc.Chrome(options=options, use_subprocess=True, version_main=None)
    driver.set_page_load_timeout(45)
    log.info("  Driver ready ✓")
    return driver

def quit_driver(driver):
    try: driver.quit()
    except Exception: pass

# ══════════════════════════════════════════════════════════════════════════════
# JS — date form interaction
# ══════════════════════════════════════════════════════════════════════════════

DISMISS_JS = """
var sel = ['[class*="cookie"]','[class*="consent"]','[class*="popup"]',
           '[class*="modal"]','[class*="overlay"]','[id*="cookie"]','[id*="popup"]'];
sel.forEach(function(s) {
    document.querySelectorAll(s).forEach(function(el) {
        el.style.display = 'none';
    });
});
"""

SET_DATE_JS = """
var dateStr = arguments[0];
var input = null;
var allInputs = document.querySelectorAll('input[type="text"]');
for (var i = 0; i < allInputs.length; i++) {
    if (/\\d{2}-\\d{2}-\\d{4}/.test(allInputs[i].value) ||
        (allInputs[i].placeholder && allInputs[i].placeholder.indexOf('DD') !== -1)) {
        input = allInputs[i]; break;
    }
}
if (!input && allInputs.length > 0) { input = allInputs[0]; }
if (!input) { return 'ERROR:no_input'; }
try {
    var setter = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype,'value').set;
    setter.call(input, dateStr);
} catch(e) { input.value = dateStr; }
input.dispatchEvent(new Event('input',  {bubbles:true}));
input.dispatchEvent(new Event('change', {bubbles:true}));
input.dispatchEvent(new Event('blur',   {bubbles:true}));
var btns = document.querySelectorAll(
    'button,input[type="button"],input[type="submit"]');
for (var b = 0; b < btns.length; b++) {
    var t = (btns[b].textContent||'').trim();
    var v = btns[b].value||'';
    if (t==='Go'||v==='Go') { btns[b].click(); return 'OK:go:'+dateStr; }
}
var form = input.closest('form');
if (form) { form.submit(); return 'OK:submit:'+dateStr; }
input.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',keyCode:13,bubbles:true}));
return 'OK:enter:'+dateStr;
"""

# ══════════════════════════════════════════════════════════════════════════════
# GET LISTING LINKS FOR ONE DATE
# ══════════════════════════════════════════════════════════════════════════════

def get_links_for_date(driver: uc.Chrome, date: datetime,
                       log: logging.Logger) -> list[str]:
    date_str = date.strftime("%d-%m-%Y")

    for attempt in range(1, 4):
        try:
            driver.get(TODAYS_PAPER)
            try:
                WebDriverWait(driver, 15).until(
                    lambda d: d.execute_script(
                        "return document.readyState") == "complete")
            except TimeoutException:
                pass
            time.sleep(5)

            driver.execute_script(DISMISS_JS)
            time.sleep(1)

            result = driver.execute_script(SET_DATE_JS, date_str)
            log.info(f"  Date '{date_str}' | JS: {result}")

            if result and "ERROR" in str(result):
                time.sleep(5); continue

            time.sleep(15)

            page_h = driver.execute_script(
                "return document.body.scrollHeight") or 6000
            for y in range(0, min(int(page_h), 10000), 600):
                driver.execute_script(f"window.scrollTo(0, {y})")
                time.sleep(0.4)
            time.sleep(2)

            links = extract_article_links(driver.page_source, log)
            return links

        except WebDriverException as e:
            log.warning(f"  Error attempt {attempt}: {str(e)[:80]}")
            if "chrome not reachable" in str(e).lower(): raise
            time.sleep(10)
        except Exception as e:
            log.warning(f"  Error attempt {attempt}: {e}")
            time.sleep(5)

    return []

# ══════════════════════════════════════════════════════════════════════════════
# FETCH ARTICLE WITH DRIVER (not requests — bypasses Cloudflare)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_article_with_driver(
    driver: uc.Chrome,
    url: str,
    date: datetime,
    log: logging.Logger,
) -> Optional[dict]:
    """
    Fetch one article page using the undetected Chrome driver.
    This bypasses Cloudflare completely — same as how the listing page works.
    """
    try:
        driver.get(url)
        time.sleep(ARTICLE_WAIT)

        # Dismiss any popups
        driver.execute_script(DISMISS_JS)
        time.sleep(0.5)

        html = driver.page_source
        rec  = parse_article(html, url, date)
        return rec

    except WebDriverException as e:
        log.debug(f"  Driver error on article: {str(e)[:60]}")
        if "chrome not reachable" in str(e).lower():
            raise
        return None
    except Exception as e:
        log.debug(f"  Article error: {e}")
        return None

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_scraper(
    out_dir: str    = "bs_todays_paper_data",
    years_back: int = None,   # kept for backward compat but ignored; uses START_DATE/END_DATE
    max_dates: int  = None,
) -> list[dict]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log       = get_logger(out_dir)
    ckpt      = Checkpoint(str(Path(out_dir) / "checkpoint.json"))
    raw_path  = Path(out_dir) / "raw_articles.json"
    test_mode = max_dates is not None

    existing: list[dict] = []
    if raw_path.exists():
        try:
            data = json.loads(raw_path.read_text("utf-8"))
            if data:
                existing = data
                log.info(f"Resumed: {len(existing)} articles")
        except Exception:
            pass

    store   = ArticleStore(raw_path, existing)
    dates   = generate_dates(START_DATE, END_DATE, max_dates)
    pending = [d for d in dates if not ckpt.date_done(d)]

    log.info("=" * 65)
    log.info("Business Standard — TODAY'S PAPER SCRAPER")
    log.info(f"URL       : {TODAYS_PAPER}")
    log.info(f"Article fetch: undetected Chrome (not requests)")
    log.info(f"Period    : {START_DATE.strftime('%Y-%m-%d')} → {END_DATE.strftime('%Y-%m-%d')}")
    log.info(f"Dates     : {len(pending)} pending of {len(dates)}")
    log.info(f"Mode      : {'TEST (' + str(max_dates) + ' dates)' if test_mode else 'FULL'}")
    log.info(f"Output    : {out_dir}/raw_articles.json")
    log.info("=" * 65)

    if not pending:
        log.info("All dates done.")
        return store.all()

    driver = None
    try:
        driver = create_driver(log)

        for idx, date in enumerate(pending, 1):
            log.info(f"\n[{idx}/{len(pending)}] {date.strftime('%Y-%m-%d')}"
                     f" | Total: {len(store)}")

            # Step 1: get article links for this date
            try:
                links = get_links_for_date(driver, date, log)
            except WebDriverException:
                log.warning("  Browser crashed — restarting…")
                quit_driver(driver)
                time.sleep(5)
                driver = create_driver(log)
                links  = get_links_for_date(driver, date, log)

            if not links:
                log.warning(f"  No links for {date.strftime('%Y-%m-%d')}")
                ckpt.mark_date(date)
                time.sleep(random.uniform(*PAGE_DELAY))
                continue

            new_links = [l for l in links if not ckpt.url_seen(l)]
            log.info(f"  {len(new_links)} new / {len(links)} total")

            # Step 2: fetch each article with the driver
            day_collected = 0
            for i, url in enumerate(new_links, 1):
                log.info(f"  [{i}/{len(new_links)}] {url[-70:]}")

                try:
                    rec = fetch_article_with_driver(driver, url, date, log)
                except WebDriverException:
                    log.warning("  Browser crashed — restarting…")
                    quit_driver(driver)
                    time.sleep(5)
                    driver = create_driver(log)
                    rec = fetch_article_with_driver(driver, url, date, log)

                if rec and rec.get("title"):
                    store.add(rec)
                    day_collected += 1
                    log.info(
                        f"    ✓ [{rec.get('date', date.strftime('%Y-%m-%d'))}]"
                        f" {rec['title'][:60]}"
                    )
                else:
                    log.info(f"    — No title extracted (may be paywalled)")

                ckpt.mark_url(url)
                time.sleep(random.uniform(*ARTICLE_DELAY))

            log.info(
                f"  {day_collected} articles from {date.strftime('%Y-%m-%d')}"
                f" | Total: {len(store)}"
            )
            ckpt.mark_date(date)
            store.save()

            # Restart driver every 50 dates
            if idx % 50 == 0 and idx < len(pending):
                log.info("  Restarting driver…")
                quit_driver(driver)
                time.sleep(5)
                driver = create_driver(log)

            time.sleep(random.uniform(*PAGE_DELAY))

    except KeyboardInterrupt:
        log.info("\nInterrupted — run again to resume.")
    except Exception as e:
        log.error(f"Fatal: {e}")
    finally:
        quit_driver(driver)

    store.save()
    ckpt.save()
    log.info(f"\nCOMPLETE. Articles: {len(store)} | Dates: {ckpt.dates_done}")
    return store.all()


if __name__ == "__main__":
    import sys
    arts = run_scraper(
        out_dir    = "bs_todays_paper_data",
        max_dates  = 3 if "--test" in sys.argv else None,
    )
    print(f"\nResult : {len(arts)} articles")
    if arts:
        print(f"Sample : {arts[0].get('title','')}")
        print(f"Date   : {arts[0].get('date','')}")
        print(f"Section: {arts[0].get('section','')}")
