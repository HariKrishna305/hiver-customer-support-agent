import os
import json
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GOLDEN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labeled_golden_set.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reply_quality_results.csv"
)


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file"
    )

client = genai.Client(api_key=api_key)

MODEL = "gemini-3.6-flash"


# --------------------------------------------------
# Generate reply
# --------------------------------------------------

def generate_reply(customer_message, brand_reply):

    prompt = f"""
You are an Amazon customer support reply assistant.

Draft a short customer-facing reply.

Customer message:
{customer_message}

Historical Amazon support reply:
{brand_reply}

Rules:
- Address the customer's issue directly.
- Use the historical reply as grounding evidence.
- Do not invent policies, refunds, delivery dates, account information,
  or actions that are not supported by the historical reply.
- Do not mention this evaluation.
- Return only the proposed customer-facing reply.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text.strip()


# --------------------------------------------------
# LLM Judge
# --------------------------------------------------

def judge_reply(
    customer_message,
    historical_reply,
    generated_reply
):

    prompt = f"""
You are evaluating an AI-generated customer support reply.

Customer message:
{customer_message}

Historical brand reply:
{historical_reply}

AI-generated reply:
{generated_reply}

Evaluate the AI reply using these criteria.

1. Relevance
Does the reply address the customer's actual issue?

2. Grounding
Is the reply supported by the historical brand reply?
Does it avoid unsupported claims?

3. Helpfulness
Does it provide an appropriate next step or useful response?

4. Professionalism
Is it clear, concise, polite, and suitable for customer support?

5. Hallucination
Does the reply introduce unsupported information?

Give each score from 1 to 5.

Return ONLY valid JSON in this exact format:

{{
    "relevance": 1,
    "grounding": 1,
    "helpfulness": 1,
    "professionalism": 1,
    "hallucination": 1,
    "overall": 1,
    "reason": "short explanation"
}}

For hallucination:
1 = severe unsupported claims
2 = significant unsupported claims
3 = minor unsupported claims
4 = mostly grounded
5 = fully grounded
"""


    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    text = response.text.strip()

    # Remove markdown code fences if Gemini returns them
    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:

        return {
            "relevance": 0,
            "grounding": 0,
            "helpfulness": 0,
            "professionalism": 0,
            "hallucination": 0,
            "overall": 0,
            "reason": "Judge returned invalid JSON."
        }


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

def main():

    print("Loading golden dataset...")

    df = pd.read_csv(GOLDEN_FILE)

    print(f"Golden examples available: {len(df)}")

    # --------------------------------------------------
    # Use a manageable evaluation subset
    # --------------------------------------------------

    evaluation_df = df.head(30).copy()

    print(
        f"Examples selected for reply evaluation: "
        f"{len(evaluation_df)}"
    )

    results = []

    print("\nGenerating and judging replies...\n")

    for index, row in evaluation_df.iterrows():

        print(
            f"Processing example "
            f"{index + 1}/{len(evaluation_df)}..."
        )

        customer_message = str(
            row["customer_message"]
        )

        historical_reply = str(
            row["brand_reply"]
        )

        try:

            generated_reply = generate_reply(
                customer_message,
                historical_reply
            )

            judge_result = judge_reply(
                customer_message,
                historical_reply,
                generated_reply
            )

            results.append({
                "customer_message": customer_message,
                "historical_reply": historical_reply,
                "generated_reply": generated_reply,
                "relevance": judge_result.get(
                    "relevance", 0
                ),
                "grounding": judge_result.get(
                    "grounding", 0
                ),
                "helpfulness": judge_result.get(
                    "helpfulness", 0
                ),
                "professionalism": judge_result.get(
                    "professionalism", 0
                ),
                "hallucination": judge_result.get(
                    "hallucination", 0
                ),
                "overall": judge_result.get(
                    "overall", 0
                ),
                "reason": judge_result.get(
                    "reason", ""
                )
            })

        except Exception as e:

            print(
                f"Error processing example "
                f"{index + 1}: {e}"
            )

        # Small delay to avoid hitting API limits
        time.sleep(1)


    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------
    # Display summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("REPLY QUALITY EVALUATION")
    print("=" * 60)

    print(
        f"\nExamples evaluated: "
        f"{len(results_df)}"
    )

    if not results_df.empty:

        metrics = [
            "relevance",
            "grounding",
            "helpfulness",
            "professionalism",
            "hallucination",
            "overall"
        ]

        print("\nAverage Scores:")

        for metric in metrics:

            print(
                f"{metric.capitalize():20s}: "
                f"{results_df[metric].mean():.2f}/5"
            )

    print(
        f"\nResults saved to:"
        f"\n{OUTPUT_FILE}"
    )

    print("=" * 60)


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    main()