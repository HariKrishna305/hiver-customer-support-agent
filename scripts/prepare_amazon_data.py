import pandas as pd
from pathlib import Path


INPUT_FILE = "data/processed/amazon_conversations.csv"
OUTPUT_FILE = "data/processed/amazon_clean.csv"


def main():

    print("Loading Amazon conversation data...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Original rows: {len(df)}")

    # Remove missing messages/replies
    df = df.dropna(
        subset=["customer_message", "brand_reply"]
    )

    # Convert to string
    df["customer_message"] = (
        df["customer_message"]
        .astype(str)
        .str.strip()
    )

    df["brand_reply"] = (
        df["brand_reply"]
        .astype(str)
        .str.strip()
    )

    # Remove empty messages
    df = df[
        (df["customer_message"] != "") &
        (df["brand_reply"] != "")
    ]

    # Remove exact duplicate conversations
    df = df.drop_duplicates(
        subset=["customer_message", "brand_reply"]
    )

    # Remove extremely short messages
    df = df[
        df["customer_message"].str.len() >= 5
    ]

    # Remove extremely short replies
    df = df[
        df["brand_reply"].str.len() >= 5
    ]

    Path("data/processed").mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("AMAZON DATA CLEANING COMPLETED")
    print("=" * 60)

    print(f"Clean rows: {len(df)}")
    print(f"Removed rows: {168814 - len(df)}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nSample:")
    print(
        df[
            ["customer_message", "brand_reply"]
        ].head(5).to_string(index=False)
    )


if __name__ == "__main__":
    main()