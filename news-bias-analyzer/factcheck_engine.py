import re


def extract_claims(article_texts):
    """
    Extract factual-looking claims
    """

    claims = []

    patterns = [

        r'\d+%',

        r'\$\d+',

        r'\d+\speople',

        r'according to',

        r'research shows',

        r'study finds'
    ]

    for article in article_texts:

        sentences = article.split(". ")

        for sentence in sentences:

            for pattern in patterns:

                if re.search(
                    pattern,
                    sentence,
                    re.IGNORECASE
                ):

                    claims.append(sentence.strip())

                    break

    return claims


def generate_factcheck_suggestions(claims):
    """
    Generate verification suggestions
    """

    suggestions = []

    for claim in claims:

        suggestions.append({

            "claim": claim,

            "verify_with": [

                "Reuters Fact Check",

                "Snopes",

                "PolitiFact",

                "Official Government Sources"
            ]
        })

    return suggestions


def analyze_fact_checks(article_texts):
    """
    Main analysis function
    """

    claims = extract_claims(article_texts)

    suggestions = generate_factcheck_suggestions(claims)

    return {

        "claims_detected": claims,

        "factcheck_suggestions": suggestions
    }


if __name__ == "__main__":

    sample_articles = [

        """
        Research shows 65% of citizens
        support the new law.
        """,

        """
        According to officials,
        $5 million will be invested next year.
        """
    ]

    result = analyze_fact_checks(sample_articles)

    print("\n===== FACT CHECK ANALYSIS =====\n")

    print(result)