from summary_engine import generate_balanced_summary
from factcheck_engine import analyze_fact_checks


sample_articles = [

    """
    Government announces new climate policy.
    Research shows 70% support among citizens.
    """,

    """
    Critics say the policy could increase taxes.
    According to experts, industries may face pressure.
    """
]


summary = generate_balanced_summary(sample_articles)

factchecks = analyze_fact_checks(sample_articles)


print("\n===== BALANCED SUMMARY =====\n")

print(summary)


print("\n===== FACT CHECK RESULTS =====\n")

print(factchecks)