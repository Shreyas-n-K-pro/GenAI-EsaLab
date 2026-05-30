import pandas as pd
from collections import Counter
import re
from google import genai
import os


# =========================
# GEMINI SETUP
# =========================

API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_AVAILABLE = False

try:

    client = genai.Client(api_key=API_KEY)

    GEMINI_AVAILABLE = True

except:

    GEMINI_AVAILABLE = False


# =========================
# TEXT CLEANING
# =========================

def clean_text(text):
    """
    Clean article text
    """

    text = str(text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# =========================
# FALLBACK SUMMARIZER
# =========================

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


# =========================
# MAIN SUMMARY FUNCTION
# =========================

def generate_balanced_summary(article_texts):
    """
    Generate balanced summary using Gemini.
    Fallback to rule-based summarizer.
    """

    cleaned_articles = [
        clean_text(text)
        for text in article_texts
    ]

    combined_text = " ".join(cleaned_articles)

    # GEMINI SUMMARY
    if GEMINI_AVAILABLE:

        try:

            prompt = f"""
            Create a short neutral news summary
            combining multiple perspectives.

            Keep it balanced and factual.

            ARTICLES:
            {combined_text[:4000]}
            """

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )

            return response.text

        except Exception:

            print("Gemini unavailable. Using fallback summarizer.")

    # FALLBACK SUMMARY
    return fallback_summary(cleaned_articles)


# =========================
# TESTING
# =========================

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