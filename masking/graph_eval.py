# Co-occurrence & confusion graph measures
import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from metric import DIMENSIONS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/train.csv")
    parser.add_argument("--truth", default="data/synthetic_test.csv")
    parser.add_argument("--pred", default="data/synthetic_preds.csv")
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv)
    truth = pd.read_csv(args.truth).set_index("image_id")
    pred = pd.read_csv(args.pred).set_index("image_id")

    print("=== Top Cross-Dimension Confusion Edges (Truth -> Pred) ===")
    for dim in DIMENSIONS:
        valid = truth[dim].notna()
        t_col = truth.loc[valid, dim].astype(str)
        p_col = pred.loc[valid, dim].astype(str)
        errors = pd.DataFrame({"truth": t_col, "pred": p_col})
        errors = errors[errors["truth"] != errors["pred"]]
        top_edges = errors.value_counts().head(5)
        print(f"\n[{dim}] ({len(errors)}/{len(t_col)} errors)")
        for (t_val, p_val), cnt in top_edges.items():
            print(f"  {t_val} -> {p_val}: {cnt}")

    # Check unseen (subject, background) pairs against training co-occurrence graph
    seen_pairs = set(zip(train["subject"].dropna(), train["background"].dropna()))
    pred_pairs = list(zip(pred["subject"], pred["background"]))
    unseen = sum(1 for p in pred_pairs if p not in seen_pairs)
    print(f"\nUnseen (subject, background) graph edges in predictions: {unseen}/{len(pred_pairs)}")

if __name__ == "__main__":
    main()