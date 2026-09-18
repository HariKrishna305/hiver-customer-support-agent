import pandas as pd
from sklearn.metrics import mean_absolute_error, cohen_kappa_score


LLM_FILE = "data/processed/reply_quality_results.csv"
HUMAN_FILE = "data/processed/human_reply_evaluation.csv"


def main():

    print("Loading evaluation files...")

    llm = pd.read_csv(LLM_FILE)
    human = pd.read_csv(HUMAN_FILE)

    # The LLM results file does not contain example_id.
    # The human file contains 5 rows corresponding to the
    # first 5 LLM-evaluated examples.
    llm = llm.head(len(human)).copy()

    # Create matching example IDs
    llm["example_id"] = range(1, len(llm) + 1)

    print(f"LLM examples available: {len(llm)}")
    print(f"Human examples: {len(human)}")

    merged = pd.merge(
        llm,
        human,
        on="example_id",
        how="inner"
    )

    print(f"Examples compared: {len(merged)}")

    dimensions = [
        "relevance",
        "grounding",
        "helpfulness",
        "professionalism",
        "hallucination"
    ]

    print("\n" + "=" * 60)
    print("LLM JUDGE vs HUMAN AGREEMENT")
    print("=" * 60)

    for dimension in dimensions:

        llm_scores = merged[dimension].astype(int)
        human_scores = merged[f"human_{dimension}"].astype(int)

        mae = mean_absolute_error(
            human_scores,
            llm_scores
        )

        kappa = cohen_kappa_score(
            human_scores,
            llm_scores
        )

        exact_match = (
            llm_scores == human_scores
        ).mean()

        print(f"\n{dimension.upper()}")
        print(f"MAE: {mae:.2f}")
        print(f"Cohen's Kappa: {kappa:.2f}")
        print(f"Exact agreement: {exact_match:.2%}")

    # Overall comparison
    human_columns = [
        "human_relevance",
        "human_grounding",
        "human_helpfulness",
        "human_professionalism",
        "human_hallucination"
    ]

    llm_columns = [
        "relevance",
        "grounding",
        "helpfulness",
        "professionalism",
        "hallucination"
    ]

    overall_human = merged[human_columns].mean(axis=1)
    overall_llm = merged[llm_columns].mean(axis=1)

    overall_mae = mean_absolute_error(
        overall_human,
        overall_llm
    )

    print("\n" + "=" * 60)
    print("OVERALL AGREEMENT")
    print("=" * 60)

    print(f"Human average: {overall_human.mean():.2f}/5")
    print(f"LLM average:   {overall_llm.mean():.2f}/5")
    print(f"Overall MAE:   {overall_mae:.2f}")

    print("=" * 60)


if __name__ == "__main__":
    main()