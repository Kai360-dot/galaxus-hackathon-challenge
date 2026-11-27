import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from metric import DIMENSIONS, VALID_CLASSES

LABEL_MAPS = {d: sorted(VALID_CLASSES[d]) for d in DIMENSIONS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-csv", default="data/synthetic_test.csv")
    parser.add_argument("--probs-npz", default="data/siglip2_synthetic_probs.npz")
    parser.add_argument("--out-json", default="checkpoints/class_multipliers.json")
    args, _ = parser.parse_known_args()

    truth_df = pd.read_csv(args.truth_csv)
    npz = np.load(args.probs_npz, allow_pickle=True)
    prob_ids = [str(x) for x in npz["image_ids"]]
    idx_lookup = {img_id: i for i, img_id in enumerate(prob_ids)}

    aligned_indices = [idx_lookup[str(img_id)] for img_id in truth_df["image_id"]]
    multipliers = {}

    for d in DIMENSIONS:
        classes = LABEL_MAPS[d]
        probs = npz[d][aligned_indices]
        y_true_raw = truth_df[d].fillna("").astype(str).str.strip().tolist()
        valid_mask = [val in classes for val in y_true_raw]
        y_true = [y_true_raw[i] for i, ok in enumerate(valid_mask) if ok]
        p_valid = probs[valid_mask]

        best_mult = np.ones(len(classes), dtype=np.float32)
        base_preds = [classes[int(i)] for i in p_valid.argmax(axis=1)]
        best_f1 = f1_score(y_true, base_preds, labels=classes, average="macro", zero_division=0)

        # Coordinate ascent over class multipliers
        for _ in range(3):
            for c_idx in range(len(classes)):
                for cand in np.linspace(0.55, 2.25, 35):
                    trial = best_mult.copy()
                    trial[c_idx] = cand
                    preds = [classes[int(i)] for i in (p_valid * trial).argmax(axis=1)]
                    score = f1_score(y_true, preds, labels=classes, average="macro", zero_division=0)
                    if score > best_f1 + 1e-5:
                        best_f1 = score
                        best_mult = trial

        multipliers[d] = {classes[i]: float(best_mult[i]) for i in range(len(classes))}
        print(f"{d:22s} | Tuned Macro F1: {best_f1:.4f}")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(multipliers, f, indent=2)
    print(f"Saved optimal Macro-F1 multipliers to {args.out_json}")


if __name__ == "__main__":
    main()