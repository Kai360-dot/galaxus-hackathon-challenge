import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from label_sanitizer import sanitize_dataframe
from metric import DIMENSIONS, VALID_CLASSES


def evaluate_predictions(truth_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    truth = sanitize_dataframe(truth_df, is_ground_truth=True).set_index("image_id")
    pred = sanitize_dataframe(pred_df, is_ground_truth=False).set_index("image_id")
    pred = pred.reindex(truth.index)

    all_class_f1 = []
    print("\n" + "=" * 62)
    print(f"{'DIMENSION':<22} | {'MACRO F1':>8} | {'SCORED ROWS':>11}")
    print("-" * 62)

    for d in DIMENSIONS:
        classes = sorted(VALID_CLASSES[d])
        valid_mask = truth[d].isin(classes)
        y_true = truth.loc[valid_mask, d].tolist()
        y_pred = pred.loc[valid_mask, d].tolist()

        per_cls = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
        dim_macro = float(np.mean(per_cls))
        print(f"{d:<22} | {dim_macro:>8.4f} | {len(y_true):>11d}")

        for cls_name, score in zip(classes, per_cls):
            all_class_f1.append(
                (d, cls_name, float(score), int((truth.loc[valid_mask, d] == cls_name).sum()))
            )

    overall_macro_f1 = float(np.mean([x[2] for x in all_class_f1]))
    print("-" * 62)
    print(f"{'OVERALL 41-CLASS F1':<22} | {overall_macro_f1:>8.4f} | {'Target >0.917':>11}")
    print("=" * 62)

    weakest = sorted(all_class_f1, key=lambda x: x[2])[:8]
    print("\nLowest 8 Classes (Focus for Multiplier / Prompt Tuning):")
    for d, cls_name, score, support in weakest:
        print(f"  - {d}.{cls_name:<24} F1={score:.4f} (n={support})")
    print()
    return overall_macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--pred", required=True)
    args, _ = parser.parse_known_args()
    evaluate_predictions(pd.read_csv(args.truth), pd.read_csv(args.pred))


if __name__ == "__main__":
    main()