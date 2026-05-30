"""
test_data_loader.py — Quick validation for the data ingestion layer
====================================================================

Run this to verify that all datasets load, merge, and export correctly.
This is NOT a pytest suite — it's a simple script for fast teammate validation.

Usage:
    python test_data_loader.py
"""

import sys
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import (
    load_allsides_bias,
    get_bias_map,
    load_huffpost_articles,
    load_multi_publisher_articles,
    load_all_data,
    get_articles_by_topic,
    get_available_topics,
    get_categories,
    get_bias_distribution,
    export_cleaned_data,
    BIAS_ORDER,
)


def test_allsides():
    """Verify AllSides bias ratings load correctly."""
    print("-" * 50)
    print("TEST 1: AllSides Bias Ratings")
    print("-" * 50)

    df = load_allsides_bias()
    assert len(df) > 0, "AllSides dataframe is empty!"
    assert "name" in df.columns, "Missing 'name' column"
    assert "bias_rating" in df.columns, "Missing 'bias_rating' column"

    # Check bias labels are valid
    valid = set(BIAS_ORDER)
    actual = set(df["bias_rating"].unique())
    assert actual.issubset(valid), f"Invalid bias labels found: {actual - valid}"

    print(f"  [OK] Loaded {len(df)} outlets")
    print(f"  [OK] Bias distribution: {df['bias_rating'].value_counts().to_dict()}")

    # Verify bias map
    bias_map = get_bias_map()
    assert isinstance(bias_map, dict)
    assert "CNN" in bias_map
    assert bias_map["Fox News"] == "Right"
    assert bias_map["Reuters"] == "Center"
    print(f"  [OK] Bias map has {len(bias_map)} entries")
    print()


def test_huffpost():
    """Verify HuffPost articles load correctly."""
    print("-" * 50)
    print("TEST 2: HuffPost Articles")
    print("-" * 50)

    df = load_huffpost_articles()
    assert len(df) > 0, "HuffPost dataframe is empty!"
    assert "headline" in df.columns, "Missing 'headline' column"
    assert "category" in df.columns, "Missing 'category' column"
    assert "short_description" in df.columns, "Missing 'short_description' column"

    print(f"  [OK] Loaded {len(df)} articles")
    print(f"  [OK] Categories: {sorted(df['category'].unique().tolist())}")

    # Test filtering
    politics = load_huffpost_articles(categories=["POLITICS"], max_per_category=5)
    assert all(politics["category"].str.upper() == "POLITICS")
    assert len(politics) <= 5
    print(f"  [OK] Category filter works (POLITICS: {len(politics)} articles)")
    print()


def test_multi_publisher():
    """Verify multi-publisher corpus loads correctly."""
    print("-" * 50)
    print("TEST 3: Multi-Publisher Corpus")
    print("-" * 50)

    df = load_multi_publisher_articles()
    assert len(df) > 0, "Multi-publisher dataframe is empty!"
    assert "topic_id" in df.columns, "Missing 'topic_id' column"
    assert "publisher" in df.columns, "Missing 'publisher' column"
    assert "article_text" in df.columns, "Missing 'article_text' column"

    topics = df["topic_id"].unique()
    print(f"  [OK] Loaded {len(df)} articles across {len(topics)} topics")
    for tid in topics:
        subset = df[df["topic_id"] == tid]
        pubs = subset["publisher"].unique().tolist()
        print(f"     > {tid}: {len(pubs)} publishers - {pubs}")
    print()


def test_unified():
    """Verify the unified dataset assembly."""
    print("-" * 50)
    print("TEST 4: Unified Dataset")
    print("-" * 50)

    df = load_all_data()
    required_cols = [
        "title", "publisher", "article_text", "category",
        "bias_label", "url", "published_date", "topic_id",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    assert len(df) > 0, "Unified dataframe is empty!"
    assert df["title"].notna().all(), "Some titles are NaN"
    assert df["bias_label"].notna().all(), "Some bias labels are NaN"

    print(f"  [OK] Shape: {df.shape}")
    print(f"  [OK] Columns: {list(df.columns)}")
    print(f"  [OK] Publishers ({df['publisher'].nunique()}): {sorted(df['publisher'].unique().tolist())}")
    print(f"  [OK] Categories ({df['category'].nunique()}): {sorted(df['category'].unique().tolist())}")
    print(f"  [OK] Bias dist: {df['bias_label'].value_counts().to_dict()}")
    print()

    # Print a sample
    print("  --- Sample (first 5 rows) ---")
    sample = df[["title", "publisher", "category", "bias_label"]].head(5)
    print(sample.to_string(index=False))
    print()


def test_query_helpers():
    """Verify query helper functions."""
    print("-" * 50)
    print("TEST 5: Query Helpers")
    print("-" * 50)

    df = load_all_data()

    # get_articles_by_topic
    budget = get_articles_by_topic("budget_2024", df)
    assert len(budget) > 0, "No articles for budget_2024!"
    print(f"  [OK] get_articles_by_topic('budget_2024'): {len(budget)} articles")

    # Verify sorted by bias order
    labels = budget["bias_label"].tolist()
    order_map = {label: i for i, label in enumerate(BIAS_ORDER)}
    indices = [order_map.get(l, 99) for l in labels]
    assert indices == sorted(indices), "Articles not sorted by bias order!"
    print(f"  [OK] Sorted by bias: {labels}")

    # get_available_topics
    topics = get_available_topics(df)
    assert len(topics) > 0
    print(f"  [OK] get_available_topics(): {len(topics)} topics")

    # get_categories
    cats = get_categories(df)
    assert len(cats) > 0
    print(f"  [OK] get_categories(): {cats}")

    # get_bias_distribution
    dist = get_bias_distribution(df)
    assert "bias_label" in dist.columns
    assert "count" in dist.columns
    print(f"  [OK] get_bias_distribution():")
    for _, row in dist.iterrows():
        print(f"     {row['bias_label']:>12}: {row['count']}")
    print()


def test_export():
    """Verify cleaned data export."""
    print("-" * 50)
    print("TEST 6: Export Cleaned Data")
    print("-" * 50)

    df = load_all_data()
    paths = export_cleaned_data(df)

    for name, path in paths.items():
        p = Path(path)
        assert p.exists(), f"Export file missing: {path}"
        size_kb = p.stat().st_size / 1024
        print(f"  [OK] {name}: {path} ({size_kb:.1f} KB)")
    print()


def main():
    print("=" * 60)
    print("  News Bias Analyzer — Data Layer Tests")
    print("=" * 60)
    print()

    try:
        test_allsides()
        test_huffpost()
        test_multi_publisher()
        test_unified()
        test_query_helpers()
        test_export()

        print("=" * 60)
        print("  ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
