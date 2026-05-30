import pandas as pd
from collections import Counter
import re

# OPTIONAL AI summarizer
try:
    from transformers import pipeline

    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

    AI_AVAILABLE = True

except:
    AI_AVAILABLE = False


def clean_text(text):
    """
    Clean article text
    """

    text = str(text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def fallback_summary(texts, max_sentences=3):
    """
    Rule-based fallback summary
    """

    combined = " ".join(texts)

    sentences = combined.split(". ")

    word_freq = Counter(combined.lower().split())

    scored_sentences = []

    for sentence in sentences:

        score = sum(
            word_freq[word.lower()]
            for word in sentence.split()
        )

        scored_sentences.append((score, sentence))

    scored_sentences = sorted(
        scored_sentences,
        reverse=True
    )

    top_sentences = [
        s[1]
        for s in scored_sentences[:max_sentences]
    ]

    return ". ".join(top_sentences)


def generate_balanced_summary(article_texts):
    """
    Generate balanced summary
    """

    cleaned_articles = [
        clean_text(text)
        for text in article_texts
    ]

    combined_text = " ".join(cleaned_articles)

    # AI summarizer
    if AI_AVAILABLE:

        try:

            result = summarizer(
                combined_text[:3000],
                max_length=120,
                min_length=40,
                do_sample=False
            )

            return result[0]["summary_text"]

        except:
            pass

    # fallback summary
    return fallback_summary(cleaned_articles)


if __name__ == "__main__":

    sample_articles = [

        """
        Government announces new climate policy
        focusing on renewable energy.
        """,

        """
        Critics argue the policy could
        increase taxes and hurt industries.
        """,

        """
        Experts believe the long-term
        environmental impact could be positive.
        """
    ]

    summary = generate_balanced_summary(sample_articles)

    print("\n===== BALANCED SUMMARY =====\n")

    print(summary)