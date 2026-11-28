"""Out-of-fold predictions over the whole train part, to tune class biases on 11,000 images instead of 600.

    python src/oof.py --data-dir ~/galaxus/data \
        --heads results/heads_512/features=24:both:1024=results/dino_448_mirrored,results/pixels \
        --folds 5 --seeds 5 --output-dir outputs/oof

The train part is split into folds of whole products. For each fold, heads are trained on the other folds and predict
the fold's images (with context, not mirrored), so every train image gets a prediction from heads that never saw it.
Class biases (an offset per class on the log-probabilities) are tuned on these out-of-fold predictions and checked on
the holdout, which neither the heads nor the biases have seen. The dev submission uses heads trained on the whole
train part, with and without the biases.

--product-mean appends to each image the mean features of all images of its product (product context).
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

import common
from ensemble import apply_biases, argmax_answers, extra_features, line, score, to_frame, tune_biases, write_submission
from heads import load_features, load_parts, predict, select, targets, train_heads


def features(spec, part, product_of, product_mean):
    """Feature rows of one part: ids, mirrored, context flags and the matrix (Apertus + helper features)."""
    features_dir, configuration, *extra = spec.split("=")
    layers_text, pooling, _ = configuration.split(":")
    ids, mirrored, context, raw, layers = load_features(Path(features_dir), part)
    x = select(raw, [layers.index(int(layer)) for layer in layers_text.split(",")], pooling)
    for extra_dir in (extra[0].split(",") if extra else []):
        x = np.hstack([x, extra_features(extra_dir, part, ids, mirrored)])
    if product_mean:  # mean over the product's images as they are (with context, not mirrored)
        products = np.array([product_of[i] for i in ids])
        plain = context & ~mirrored
        means = {p: x[plain & (products == p)].mean(0) for p in set(products[plain])}
        x = np.hstack([x, np.stack([means.get(p, x[k]) for k, p in enumerate(products)])])
    return ids, mirrored, context, x


def mean_over_seeds(x_train, y, outputs, hidden, epochs, seeds, device):
    sums = [None] * len(outputs)
    for seed in range(seeds):
        heads, mean, std = train_heads(x_train, y, hidden, epochs, device, common.SEED + seed)
        for k, x in enumerate(outputs):
            p = predict(heads, mean, std, x, device)
            sums[k] = p if sums[k] is None else {d: sums[k][d] + p[d] for d in p}
        del heads
        torch.cuda.empty_cache()
    return [{d: v / seeds for d, v in s.items()} for s in sums]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--heads", required=True, help="FEATURES_DIR=LAYERS:POOLING:HIDDEN[=EXTRA_DIR[,EXTRA_DIR...]]")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--product-mean", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oof"))
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    parts = load_parts(arguments.data_dir)
    metric = common.load_metric(arguments.data_dir)
    product_of = {i: p for part in parts.values() for i, p in zip(part["image_id"], part["product_id"])}
    hidden = int(arguments.heads.split("=")[1].split(":")[2])
    train_ids, train_mirrored, train_context, x_train = features(arguments.heads, "train", product_of, arguments.product_mean)
    hold_ids, hold_mirrored, hold_context, x_hold = features(arguments.heads, "holdout", product_of, arguments.product_mean)
    dev_ids, _, _, x_dev = features(arguments.heads, "dev", product_of, arguments.product_mean)
    keep = hold_context & ~hold_mirrored
    hold_ids, x_hold = hold_ids[keep], x_hold[keep]
    y = targets(parts["train"], train_ids, train_mirrored)
    print(f"train rows {len(x_train)} x {x_train.shape[1]} | holdout {len(x_hold)} | dev {len(x_dev)}", flush=True)

    # Out-of-fold predictions over train products.
    products = sorted(set(product_of[i] for i in train_ids))
    random.Random(common.SEED).shuffle(products)
    fold_of = {p: k % arguments.folds for k, p in enumerate(products)}
    row_fold = np.array([fold_of[product_of[i]] for i in train_ids])
    plain = train_context & ~train_mirrored
    oof = {}
    for k in range(arguments.folds):
        fit = row_fold != k
        target = plain & (row_fold == k)
        y_fit = {d: v[torch.from_numpy(fit)] for d, v in y.items()}
        (p,) = mean_over_seeds(x_train[fit], y_fit, [x_train[target]], hidden, arguments.epochs, arguments.seeds, arguments.device)
        oof.update(to_frame(train_ids[target], p))
        print(f"fold {k + 1}/{arguments.folds}: {target.sum()} images predicted", flush=True)
    np.save(arguments.output_dir / "oof_probabilities.npy", oof, allow_pickle=True)
    oof_rows = parts["train"][parts["train"]["image_id"].isin(oof)].reset_index(drop=True)
    print(line("out-of-fold (train)", score(metric, oof_rows, oof)))

    # Heads on the whole train part for the holdout check and the dev submission.
    hold_p, dev_p = mean_over_seeds(x_train, y, [x_hold, x_dev], hidden, arguments.epochs, arguments.seeds, arguments.device)
    hold = to_frame(hold_ids, hold_p)
    dev = to_frame(dev_ids, dev_p)
    hold_rows = parts["holdout"][parts["holdout"]["image_id"].isin(hold)].reset_index(drop=True)
    biases = tune_biases(oof, oof_rows)
    print(line("out-of-fold + biases", score(metric, oof_rows, apply_biases(oof, biases))))
    print(line("holdout", score(metric, hold_rows, hold)))
    print(line("holdout + OOF biases", score(metric, hold_rows, apply_biases(hold, biases))))
    print("  biases: " + " | ".join(f"{d[:8]} " + ",".join(f"{c}={b:+.2f}" for c, b in zip(common.CLASSES[d], biases[d]) if b)
                                    for d in common.DIMENSIONS))
    np.save(arguments.output_dir / "biases.npy", biases, allow_pickle=True)
    np.save(arguments.output_dir / "holdout_probabilities.npy", hold, allow_pickle=True)
    np.save(arguments.output_dir / "dev_probabilities.npy", dev, allow_pickle=True)
    write_submission(arguments.data_dir, argmax_answers(dev), arguments.output_dir / "submission_plain.csv")
    write_submission(arguments.data_dir, argmax_answers(apply_biases(dev, biases)), arguments.output_dir / "submission_biases.csv")


if __name__ == "__main__":
    main()
