import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from label_sanitizer import sanitize_dataframe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", default="data/train.csv")
    parser.add_argument("--train-out", default="data/real_train.csv")
    parser.add_argument("--val-out", default="data/real_val.csv")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args()

    df = sanitize_dataframe(pd.read_csv(args.in_csv), is_ground_truth=True)

    # Stratify on joint text_classification + perspective to preserve rare tail classes
    strat_key = df["text_classification"].fillna("none") + "__" + df["perspective"].fillna("none")
    counts = strat_key.value_counts()
    strat_key = strat_key.where(strat_key.map(counts) >= 2, "rare_bucket")

    train_df, val_df = train_test_split(
        df, test_size=args.val_frac, random_state=args.seed, stratify=strat_key
    )
    train_df.to_csv(args.train_out, index=False)
    val_df.to_csv(args.val_out, index=False)
    print(f"Created real split: {len(train_df)} train rows -> {args.train_out} | {len(val_df)} val rows -> {args.val_out}")


if __name__ == "__main__":
    main()