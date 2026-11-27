import argparse
import json
import pandas as pd
from metric import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", default="data/dev.csv")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--id-col", default="image_id")
    args = parser.parse_args()

    sol = pd.read_csv(args.truth)
    sub = pd.read_csv(args.pred)
    results = evaluate(sol, sub, args.id_col)

    print(f"Overall Macro F1 (Leaderboard): {results['overall']['f1_macro']:.5f}")
    print(f"Overall Weighted F1:            {results['overall']['f1_weighted']:.5f}")
    print("-" * 55)
    for dim, m in results.items():
        if dim == "overall":
            continue
        print(f"{dim:<22} | Macro F1: {m['f1_macro']:.5f} | Support: {m['support']}")
        worst = sorted(m["per_class"].items(), key=lambda x: x[1]["f1"])[:3]
        for cls_name, cm in worst:
            print(f"   worst: {cls_name:<22} F1={cm['f1']:.3f} (P={cm['precision']:.2f}, R={cm['recall']:.2f}, n={cm['support']})")

if __name__ == "__main__":
    main()