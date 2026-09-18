import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_FILE = "data/processed/labeled_golden_set.csv"

TRAIN_FILE = "data/processed/train.csv"
TEST_FILE = "data/processed/test.csv"


def split_dataset():

    print("Loading labeled dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Total examples: {len(df)}")

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["expected_intent"]
    )

    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print(f"Training examples: {len(train_df)}")
    print(f"Testing examples: {len(test_df)}")

    print("\nTraining intent distribution:")
    print(train_df["expected_intent"].value_counts())

    print("\nTesting intent distribution:")
    print(test_df["expected_intent"].value_counts())

    print("\nDataset split completed successfully.")


if __name__ == "__main__":
    split_dataset()