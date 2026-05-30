import json

from data_loader import (
    load_all_data,
    get_articles_by_topic,
    get_available_topics
)

# -----------------------------------
# Bias score mapping
# -----------------------------------

BIAS_SCORE = {
    "Left": -2,
    "Lean Left": -1,
    "Center": 0,
    "Lean Right": 1,
    "Right": 2
}

# -----------------------------------
# Bias explanations
# -----------------------------------

BIAS_EXPLANATIONS = {
    "Left":
        "Progressive framing and social-policy emphasis.",

    "Lean Left":
        "Moderately progressive framing.",

    "Center":
        "Attempts balanced reporting.",

    "Lean Right":
        "Moderately conservative framing.",

    "Right":
        "Conservative framing and traditional-policy emphasis."
}


# -----------------------------------
# Get numerical score
# -----------------------------------

def get_bias_score(label):

    return BIAS_SCORE.get(label, 0)


# -----------------------------------
# Explain bias label
# -----------------------------------

def explain_bias(label):

    return BIAS_EXPLANATIONS.get(
        label,
        "No explanation available."
    )


# -----------------------------------
# Missing angle detection
# -----------------------------------

def detect_missing_angle(topic_df):

    labels = topic_df["bias_label"].tolist()

    if "Left" not in labels:
        return "Progressive viewpoint missing."

    if "Right" not in labels:
        return "Conservative viewpoint missing."

    return "Multiple viewpoints represented."


# -----------------------------------
# Perspective comparison
# -----------------------------------

def perspective_difference(topic_df):

    viewpoints = []

    for _, row in topic_df.iterrows():

        viewpoints.append(
            f"{row['publisher']} emphasizes: "
            f"{row['title']}"
        )

    return viewpoints


# -----------------------------------
# Main comparison logic
# -----------------------------------

def compare_topic(topic_id):

    df = load_all_data()

    topic_df = get_articles_by_topic(
        topic_id,
        df
    )

    results = []

    for _, row in topic_df.iterrows():

        bias_label = row["bias_label"]

        results.append({

            "publisher":
                row["publisher"],

            "title":
                row["title"],

            "bias_label":
                bias_label,

            "bias_score":
                get_bias_score(
                    bias_label
                ),

            "bias_explanation":
                explain_bias(
                    bias_label
                )
        })

    return {

        "topic_id": topic_id,

        "articles": results,

        "missing_angle":
            detect_missing_angle(
                topic_df
            ),

        "perspective_difference":
            perspective_difference(
                topic_df
            )
    }


# -----------------------------------
# Demo Run
# -----------------------------------

if __name__ == "__main__":

    topics = get_available_topics()

    print("\nAVAILABLE TOPICS:\n")
    print(topics)

    selected_topic = topics.iloc[0]["topic_id"]

    print(
        f"\nRunning analysis for: "
        f"{selected_topic}\n"
    )

    result = compare_topic(
        selected_topic
    )

    from pprint import pprint

    pprint(result)

    # Save output JSON

    with open(
        "sample_bias_output.json",
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=4
        )

    print(
        "\nSaved output to "
        "sample_bias_output.json"
    )