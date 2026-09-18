import pandas as pd
from pathlib import Path


# --------------------------------------------------
# File paths
# --------------------------------------------------

INPUT_FILE = "data/archive/twcs/twcs.csv"
OUTPUT_FILE = "data/customer_brand_pairs.csv"


def create_pairs():

    print("Loading dataset...")

    # --------------------------------------------------
    # Load required columns only
    # --------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE,
        usecols=[
            "tweet_id",
            "inbound",
            "text",
            "in_response_to_tweet_id"
        ]
    )

    print(f"Original rows: {len(df):,}")

    # --------------------------------------------------
    # Remove rows without text
    # --------------------------------------------------

    df = df.dropna(subset=["text"])

    # Convert text to string and remove extra spaces
    df["text"] = df["text"].astype(str).str.strip()

    # Remove empty messages
    df = df[df["text"] != ""]

    print(f"Rows after cleaning: {len(df):,}")

    # --------------------------------------------------
    # Create lookup:
    #
    # tweet_id -> tweet information
    # --------------------------------------------------

    tweet_lookup = df.set_index("tweet_id").to_dict("index")

    pairs = []

    # --------------------------------------------------
    # Find customer message + brand reply
    # --------------------------------------------------

    for _, row in df.iterrows():

        # Brand tweets have inbound = False
        # So skip customer tweets
        if row["inbound"] == True:
            continue

        # ID of the tweet this brand reply is responding to
        parent_id = row["in_response_to_tweet_id"]

        # Skip if there is no parent tweet
        if pd.isna(parent_id):
            continue

        # Convert ID to integer
        parent_id = int(parent_id)

        # Find the original tweet
        parent = tweet_lookup.get(parent_id)

        # Skip if original tweet cannot be found
        if parent is None:
            continue

        # Original tweet must be from the customer
        if parent["inbound"] != True:
            continue

        # --------------------------------------------------
        # Create conversation pair
        # --------------------------------------------------

        customer_message = parent["text"]
        brand_reply = row["text"]

        pairs.append({
            "customer_tweet_id": parent_id,
            "brand_tweet_id": row["tweet_id"],
            "customer_message": customer_message,
            "brand_reply": brand_reply
        })

    # --------------------------------------------------
    # Convert pairs into DataFrame
    # --------------------------------------------------

    result = pd.DataFrame(pairs)

    # --------------------------------------------------
    # Remove duplicate conversations
    # --------------------------------------------------

    if not result.empty:

        result = result.drop_duplicates(
            subset=[
                "customer_message",
                "brand_reply"
            ]
        )

    # --------------------------------------------------
    # Make sure data folder exists
    # --------------------------------------------------

    Path("data").mkdir(
        exist_ok=True
    )

    # --------------------------------------------------
    # Save the cleaned conversation pairs
    # --------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print(
        f"\nConversation pairs created: {len(result):,}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    # --------------------------------------------------
    # Display sample conversations
    # --------------------------------------------------

    if not result.empty:

        print("\nSample conversations:\n")

        print(
            result[
                [
                    "customer_message",
                    "brand_reply"
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    else:

        print(
            "\nNo conversation pairs were created."
        )


# --------------------------------------------------
# Run the function
# --------------------------------------------------

if __name__ == "__main__":
    create_pairs()