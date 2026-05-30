"""
data_loader.py — Data ingestion layer for the News Bias Analyzer MVP
=====================================================================

This module handles:
  1. Loading the AllSides Media Bias Dataset (source → bias label mapping)
  2. Loading the HuffPost News Category Dataset (articles with categories)
  3. Loading the multi-publisher article corpus (same topic, different outlets)
  4. Cleaning and normalizing all data into a unified DataFrame format
  5. Exporting a cleaned, demo-ready subset to data/cleaned/

Unified output schema:
  ┌──────────────────┬───────────┬─────────────────────────────────────────────┐
  │ Column           │ Type      │ Description                                 │
  ├──────────────────┼───────────┼─────────────────────────────────────────────┤
  │ title            │ str       │ Headline / article title                    │
  │ publisher        │ str       │ News outlet name                            │
  │ article_text     │ str       │ Body or short description of the article    │
  │ category         │ str       │ Topic category (POLITICS, TECH, etc.)       │
  │ bias_label       │ str       │ AllSides-style bias (Left → Right)          │
  │ url              │ str       │ Original article URL                        │
  │ published_date   │ str       │ Publication date (YYYY-MM-DD)               │
  │ topic_id         │ str/NaN   │ Shared topic ID for multi-publisher stories │
  └──────────────────┴───────────┴─────────────────────────────────────────────┘

Usage:
  from data_loader import load_all_data, get_bias_map, get_articles_by_topic
  df = load_all_data()
  bias_map = get_bias_map()
  topic_df = get_articles_by_topic("budget_2024")

Author: Shreyas N K (shreyasnkulkarnicr7@gmail.com)
Project: GenAI-EsaLab / News Bias Analyzer MVP
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Resolve project root relative to this file's location
_MODULE_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = _MODULE_DIR / "data" / "raw"
DATA_CLEANED_DIR = _MODULE_DIR / "data" / "cleaned"

# File names (configurable if the user drops in the real datasets)
ALLSIDES_FILE = "allsides_bias_ratings.csv"
HUFFPOST_FILE = "huffpost_news_full.json"
MULTI_PUB_FILE = "multi_publisher_articles.json"

# Canonical bias labels (ordered left-to-right)
BIAS_ORDER = ["Left", "Lean Left", "Center", "Lean Right", "Right"]

# Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# 1. Load AllSides Bias Ratings
# ---------------------------------------------------------------------------

def load_allsides_bias(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load the AllSides Media Bias ratings CSV.

    Returns a DataFrame with columns:
        name, bias_rating, agree, disagree, agree_ratio, allsides_page

    The 'name' column is the publisher/outlet name, and 'bias_rating' is one
    of: Left, Lean Left, Center, Lean Right, Right.
    """
    fp = Path(filepath) if filepath else DATA_RAW_DIR / ALLSIDES_FILE
    if not fp.exists():
        raise FileNotFoundError(
            f"AllSides bias file not found at {fp}. "
            "Download it or place allsides_bias_ratings.csv in data/raw/."
        )

    df = pd.read_csv(fp)

    # Normalize column names to snake_case
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Ensure 'name' and 'bias_rating' columns exist
    required = {"name", "bias_rating"}
    missing = required - set(df.columns)
    if missing:
        # Try common alternative column names
        rename_map = {}
        if "source" in df.columns and "name" not in df.columns:
            rename_map["source"] = "name"
        if "bias" in df.columns and "bias_rating" not in df.columns:
            rename_map["bias"] = "bias_rating"
        if rename_map:
            df = df.rename(columns=rename_map)

    # Validate bias labels
    valid_labels = set(BIAS_ORDER)
    df["bias_rating"] = df["bias_rating"].str.strip()
    invalid = df[~df["bias_rating"].isin(valid_labels)]
    if not invalid.empty:
        logger.warning(
            f"Found {len(invalid)} rows with non-standard bias labels: "
            f"{invalid['bias_rating'].unique().tolist()}"
        )

    logger.info(f"Loaded {len(df)} bias ratings from {fp.name}")
    return df


def get_bias_map(filepath: Optional[str] = None) -> dict:
    """
    Return a dict mapping publisher name → bias_rating.

    Example: {"CNN": "Lean Left", "Fox News": "Right", ...}
    This is the primary lookup used by bias_engine.py.
    """
    df = load_allsides_bias(filepath)
    bias_map = dict(zip(df["name"].str.strip(), df["bias_rating"].str.strip()))
    logger.info(f"Built bias map with {len(bias_map)} outlets")
    return bias_map


# ---------------------------------------------------------------------------
# 2. Load HuffPost News Category Dataset
# ---------------------------------------------------------------------------

def load_huffpost_articles(
    filepath: Optional[str] = None,
    max_per_category: Optional[int] = None,
    categories: Optional[list] = None,
) -> pd.DataFrame:
    """
    Load the HuffPost News Category Dataset (JSON lines format).

    Parameters:
        filepath:         Path to the JSON file (defaults to data/raw/ sample).
        max_per_category: Limit articles per category for demo performance.
        categories:       Filter to specific categories (e.g., ["POLITICS", "TECHNOLOGY"]).

    Returns a DataFrame with columns:
        headline, short_description, category, authors, link, date
    """
    fp = Path(filepath) if filepath else DATA_RAW_DIR / HUFFPOST_FILE
    if not fp.exists():
        raise FileNotFoundError(
            f"HuffPost dataset not found at {fp}. "
            "Download it or place the JSON file in data/raw/."
        )

    # The HuffPost dataset is JSON-lines (one JSON object per line)
    records = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    df = pd.DataFrame(records)

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Filter categories if specified
    if categories:
        categories_upper = [c.upper() for c in categories]
        df = df[df["category"].str.upper().isin(categories_upper)]

    # Limit per category for demo size
    if max_per_category:
        # Use a selection approach that preserves all columns including 'category'
        idx = (
            df.groupby("category", group_keys=False)
            .apply(lambda g: g.head(max_per_category))
            .index
        )
        df = df.loc[idx].reset_index(drop=True)

    logger.info(
        f"Loaded {len(df)} HuffPost articles across "
        f"{df['category'].nunique()} categories from {fp.name}"
    )
    return df


# ---------------------------------------------------------------------------
# 3. Load Multi-Publisher Article Corpus
# ---------------------------------------------------------------------------

def load_multi_publisher_articles(
    filepath: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the multi-publisher article corpus (same topic, many outlets).

    This is the primary dataset for the perspective comparison feature.
    Each topic has articles from multiple publishers with different bias.

    Returns a flat DataFrame with one row per article, including topic_id
    and topic columns for grouping.
    """
    fp = Path(filepath) if filepath else DATA_RAW_DIR / MULTI_PUB_FILE
    if not fp.exists():
        raise FileNotFoundError(
            f"Multi-publisher corpus not found at {fp}. "
            "Place multi_publisher_articles.json in data/raw/."
        )

    with open(fp, "r", encoding="utf-8") as f:
        topics = json.load(f)

    rows = []
    for topic in topics:
        topic_id = topic["topic_id"]
        topic_name = topic["topic"]
        category = topic["category"]
        for article in topic["articles"]:
            rows.append({
                "topic_id": topic_id,
                "topic": topic_name,
                "category": category,
                "title": article["title"],
                "publisher": article["publisher"],
                "article_text": article["article_text"],
                "url": article.get("url", ""),
                "published_date": article.get("published_date", ""),
            })

    df = pd.DataFrame(rows)
    logger.info(
        f"Loaded {len(df)} multi-publisher articles across "
        f"{df['topic_id'].nunique()} topics"
    )
    return df


# ---------------------------------------------------------------------------
# 4. Unified Data Assembly
# ---------------------------------------------------------------------------

def _normalize_publisher_name(name: str) -> str:
    """Normalize publisher names for consistent bias map lookups."""
    # Handle common variations
    name = name.strip()
    aliases = {
        "AP": "Associated Press",
        "NYT": "The New York Times",
        "WaPo": "The Washington Post",
        "WSJ": "The Wall Street Journal",
    }
    return aliases.get(name, name)


def load_all_data(
    huffpost_max_per_cat: Optional[int] = None,
    huffpost_categories: Optional[list] = None,
) -> pd.DataFrame:
    """
    Load and merge all datasets into one unified DataFrame.

    Steps:
      1. Load AllSides bias map
      2. Load HuffPost articles → assign bias via publisher (HuffPost = Left)
      3. Load multi-publisher articles → assign bias via publisher lookup
      4. Combine into unified schema

    Returns DataFrame with columns:
      title, publisher, article_text, category, bias_label,
      url, published_date, topic_id
    """
    # --- Bias map ---
    bias_map = get_bias_map()

    # --- HuffPost articles ---
    hp_df = load_huffpost_articles(
        max_per_category=huffpost_max_per_cat,
        categories=huffpost_categories,
    )
    hp_unified = pd.DataFrame({
        "title": hp_df["headline"],
        "publisher": "HuffPost",
        "article_text": hp_df["short_description"],
        "category": hp_df["category"].str.upper(),
        "bias_label": bias_map.get("HuffPost", "Left"),
        "url": hp_df["link"],
        "published_date": hp_df["date"],
        "topic_id": pd.NA,  # HuffPost articles don't have shared topic IDs
    })

    # --- Multi-publisher articles ---
    mp_df = load_multi_publisher_articles()
    mp_df["publisher_norm"] = mp_df["publisher"].apply(_normalize_publisher_name)
    mp_df["bias_label"] = mp_df["publisher_norm"].map(bias_map)

    # Fill unmapped publishers with "Unknown"
    unmapped = mp_df[mp_df["bias_label"].isna()]["publisher"].unique()
    if len(unmapped) > 0:
        logger.warning(f"No bias rating found for publishers: {unmapped.tolist()}")
    mp_df["bias_label"] = mp_df["bias_label"].fillna("Unknown")

    mp_unified = pd.DataFrame({
        "title": mp_df["title"],
        "publisher": mp_df["publisher"],
        "article_text": mp_df["article_text"],
        "category": mp_df["category"].str.upper(),
        "bias_label": mp_df["bias_label"],
        "url": mp_df["url"],
        "published_date": mp_df["published_date"],
        "topic_id": mp_df["topic_id"],
    })

    # --- Combine ---
    unified = pd.concat([hp_unified, mp_unified], ignore_index=True)

    # Clean up
    unified["published_date"] = pd.to_datetime(
        unified["published_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    unified["category"] = unified["category"].str.upper().str.strip()
    unified["bias_label"] = unified["bias_label"].str.strip()

    logger.info(
        f"Unified dataset: {len(unified)} total articles, "
        f"{unified['publisher'].nunique()} publishers, "
        f"{unified['category'].nunique()} categories"
    )
    return unified


# ---------------------------------------------------------------------------
# 5. Query Helpers (for downstream modules)
# ---------------------------------------------------------------------------

def get_articles_by_topic(topic_id: str, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Get all articles for a specific topic_id from the multi-publisher corpus.

    Parameters:
        topic_id: The shared topic identifier (e.g., "budget_2024")
        df:       Pre-loaded unified DataFrame (loads fresh if None)

    Returns filtered DataFrame sorted by bias order (Left → Right).
    """
    if df is None:
        df = load_all_data()

    topic_df = df[df["topic_id"] == topic_id].copy()

    if topic_df.empty:
        logger.warning(f"No articles found for topic_id='{topic_id}'")
        return topic_df

    # Sort by bias order: Left → Lean Left → Center → Lean Right → Right
    bias_sort = {label: i for i, label in enumerate(BIAS_ORDER)}
    topic_df["_sort"] = topic_df["bias_label"].map(bias_sort).fillna(99)
    topic_df = topic_df.sort_values("_sort").drop(columns=["_sort"])

    return topic_df.reset_index(drop=True)


def get_available_topics(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    List all available topic_ids with their names, categories, and article counts.

    Returns a summary DataFrame useful for UI dropdowns.
    """
    if df is None:
        df = load_all_data()

    topics = (
        df[df["topic_id"].notna()]
        .groupby("topic_id")
        .agg(
            topic=("topic_id", "first"),  # Will be overridden below
            category=("category", "first"),
            num_articles=("title", "count"),
            publishers=("publisher", lambda x: ", ".join(sorted(x.unique()))),
            bias_labels=("bias_label", lambda x: ", ".join(sorted(x.unique()))),
        )
        .reset_index()
    )

    # Get topic names from multi-publisher corpus
    mp_df = load_multi_publisher_articles()
    topic_names = dict(zip(mp_df["topic_id"], mp_df["topic"]))
    topics["topic"] = topics["topic_id"].map(topic_names)

    return topics


def get_categories(df: Optional[pd.DataFrame] = None) -> list:
    """Return a sorted list of all unique categories in the dataset."""
    if df is None:
        df = load_all_data()
    return sorted(df["category"].dropna().unique().tolist())


def get_bias_distribution(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Return bias label distribution across all articles.
    Useful for viz.py charts.
    """
    if df is None:
        df = load_all_data()

    dist = (
        df["bias_label"]
        .value_counts()
        .reindex(BIAS_ORDER + ["Unknown"], fill_value=0)
        .reset_index()
    )
    dist.columns = ["bias_label", "count"]
    return dist


# ---------------------------------------------------------------------------
# 6. Export Cleaned Data
# ---------------------------------------------------------------------------

def export_cleaned_data(
    df: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Save cleaned datasets to the data/cleaned/ directory.

    Exports:
      - unified_articles.csv   (full unified dataset)
      - bias_map.json          (publisher → bias lookup)
      - topics_summary.csv     (available topics for the UI)

    Returns dict with file paths.
    """
    if df is None:
        df = load_all_data()

    out_dir = Path(output_dir) if output_dir else DATA_CLEANED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # 1. Unified articles CSV
    articles_path = out_dir / "unified_articles.csv"
    df.to_csv(articles_path, index=False, encoding="utf-8")
    paths["unified_articles"] = str(articles_path)
    logger.info(f"Exported {len(df)} articles → {articles_path}")

    # 2. Bias map JSON
    bias_map = get_bias_map()
    bias_path = out_dir / "bias_map.json"
    with open(bias_path, "w", encoding="utf-8") as f:
        json.dump(bias_map, f, indent=2, ensure_ascii=False)
    paths["bias_map"] = str(bias_path)
    logger.info(f"Exported bias map ({len(bias_map)} outlets) → {bias_path}")

    # 3. Topics summary
    topics = get_available_topics(df)
    topics_path = out_dir / "topics_summary.csv"
    topics.to_csv(topics_path, index=False, encoding="utf-8")
    paths["topics_summary"] = str(topics_path)
    logger.info(f"Exported {len(topics)} topics → {topics_path}")

    return paths


# ---------------------------------------------------------------------------
# 7. Main (run directly to build cleaned data)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  News Bias Analyzer — Data Loader")
    print("=" * 70)

    # Load everything
    df = load_all_data()

    print(f"\n[DATA] Unified Dataset Shape: {df.shape}")
    print(f"[DATA] Publishers: {df['publisher'].nunique()}")
    print(f"[DATA] Categories: {df['category'].nunique()}")
    print(f"[DATA] Bias Labels: {df['bias_label'].value_counts().to_dict()}")

    print("\n--- Sample rows ---")
    print(df[["title", "publisher", "category", "bias_label"]].head(10).to_string(index=False))

    print("\n--- Available Topics (multi-publisher) ---")
    topics = get_available_topics(df)
    print(topics.to_string(index=False))

    print("\n--- Example: Articles on 'budget_2024' ---")
    budget_articles = get_articles_by_topic("budget_2024", df)
    for _, row in budget_articles.iterrows():
        print(f"  [{row['bias_label']:>10}] {row['publisher']}: {row['title']}")

    # Export cleaned files
    print("\n--- Exporting cleaned data ---")
    paths = export_cleaned_data(df)
    for name, path in paths.items():
        print(f"  [OK] {name}: {path}")

    print("\n[DONE] Data layer ready! Other modules can now import from data_loader.py")
