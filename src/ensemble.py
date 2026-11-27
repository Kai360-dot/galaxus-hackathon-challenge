"""Combine the class probabilities of several models: heads on saved features and LoRA predictions.

    python src/ensemble.py --data-dir ~/galaxus/data \
        --heads vague=results/heads_512/features=24:both:1024 \
        --heads specific=results/heads_512_specific/features=16:both:1024 \
        --probs lora=results/lora/predictions/holdout_context.jsonl=results/lora/predictions/dev_context.jsonl \
        --output-dir outputs/ensemble

Each heads source is trained with several seeds and averaged. Scored on the holdout (with context): every source
alone, the plain mean, and per-dimension weights found on the holdout. Two-fold cross-validation over the holdout
products estimates what the weight search is worth on images it did not see. The dev submission uses the plain mean
or, with --tuned, the weights found on the whole holdout.
"""

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import common
from heads import answers_from, load_features, load_parts, predict, select, targets, train_heads


def heads_probabilities(spec, parts, seeds, epochs, device):
    """Holdout (with context) and dev probabilities of heads trained on saved features, averaged over seeds."""
    features_dir, configuration = spec.split("=")
    features_dir = Path(features_dir)
    layers_text, pooling, hidden = configuration.split(":")
    train_ids, train_mirrored, _, train_features, layers = load_features(features_dir, "train")
    holdout_ids, _, holdout_context, holdout_features, _ = load_features(features_dir, "holdout")
    dev_ids, _, _, dev_features, _ = load_features(features_dir, "dev")
    holdout_ids, holdout_features = holdout_ids[holdout_context], holdout_features[holdout_context]
    indices = [layers.index(int(layer)) for layer in layers_text.split(",")]
    y = targets(parts["train"], train_ids, train_mirrored)
    x_train = select(train_features, indices, pooling)
    sums = {"holdout": None, "dev": None}
    for seed in range(seeds):
        heads, mean, std = train_heads(x_train, y, int(hidden), epochs, device, common.SEED + seed)
        for part, x in (("holdout", select(holdout_features, indices, pooling)),
                        ("dev", select(dev_features, indices, pooling))):
            p = predict(heads, mean, std, x, device)
            sums[part] = p if sums[part] is None else {d: sums[part][d] + p[d] for d in p}
    return {
        "holdout": to_frame(holdout_ids, {d: v / seeds for d, v in sums["holdout"].items()}),
        "dev": to_frame(dev_ids, {d: v / seeds for d, v in sums["dev"].items()}),
    }


def to_frame(image_ids, probabilities):
    """image_id -> dimension -> probability array in common.CLASSES order."""
    return {image_id: {d: probabilities[d][i] for d in common.DIMENSIONS} for i, image_id in enumerate(image_ids)}


def jsonl_probabilities(spec):
    """Probabilities from prediction files with {"image_id", "probabilities": {dimension: {class: p}}} lines."""
    holdout_path, dev_path = spec.split("=")
    result = {}
    for part, path in (("holdout", holdout_path), ("dev", dev_path)):
        result[part] = {}
        if not path or not Path(path).exists():
            continue
        for line in Path(path).read_text().splitlines():
            record = json.loads(line)
            result[part][record["image_id"]] = {
                d: np.array([record["probabilities"][d][name] for name in common.CLASSES[d]])
                for d in common.DIMENSIONS
            }
    return result


def combine(sources, weights, image_ids):
    """Per image and dimension the weighted mean of the sources' probabilities."""
    combined = {}
    for image_id in image_ids:
        combined[image_id] = {}
        for d in common.DIMENSIONS:
            total = sum(weights[d][name] * sources[name][image_id][d] for name in sources)
            combined[image_id][d] = total / max(sum(weights[d].values()), 1e-9)
    return combined


def argmax_answers(probabilities):
    return {i: {d: common.CLASSES[d][int(p[d].argmax())] for d in common.DIMENSIONS} for i, p in probabilities.items()}


def score(metric, rows, probabilities):
    return common.score_rows(metric, rows, argmax_answers(probabilities), {})


def search_weights(metric, rows, sources, step):
    """Per dimension the weights (on a grid that sums to 1) with the best macro F1 on these rows."""
    names = list(sources)
    grid = [w for w in itertools.product(np.arange(0, 1 + 1e-9, step), repeat=len(names)) if abs(sum(w) - 1) < 1e-6]
    ids = list(rows["image_id"])
    best = {d: (-1.0, None) for d in common.DIMENSIONS}
    for w in grid:
        weights = {d: dict(zip(names, w)) for d in common.DIMENSIONS}
        metrics = score(metric, rows, combine(sources, weights, ids))
        for d in common.DIMENSIONS:
            if d in metrics and metrics[d]["f1_macro"] > best[d][0]:
                best[d] = (metrics[d]["f1_macro"], dict(zip(names, w)))
    return {d: best[d][1] for d in common.DIMENSIONS}


def scored_labels(rows, d):
    """The human labels of the scored cells of one dimension, None elsewhere (as metric.py scores them)."""
    scored = (rows[f"{d}_source"] == "human") & (rows[f"{d}_relevant"].str.lower() == "true")
    return np.where(scored, rows[d], None)


def macro_f1(labels, predictions):
    """Macro F1 over the classes in the labels plus the predicted ones, on the scored cells (as metric.py)."""
    keep = labels != None  # noqa: E711, elementwise
    labels, predictions = labels[keep], predictions[keep]
    classes = set(labels) | set(predictions)
    f1 = []
    for c in classes:
        tp = np.sum((predictions == c) & (labels == c))
        fp = np.sum((predictions == c) & (labels != c))
        fn = np.sum((predictions != c) & (labels == c))
        f1.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(f1)) if f1 else 0.0


def tune_biases(probabilities, rows, grid=np.arange(-2.0, 2.01, 0.25), passes=2):
    """Per dimension an additive offset per class on the log-probabilities, by coordinate ascent on macro F1."""
    ids = list(rows["image_id"])
    biases = {}
    for d in common.DIMENSIONS:
        labels = scored_labels(rows, d)
        log_p = np.log(np.stack([probabilities[i][d] for i in ids]) + 1e-9)
        classes = np.array(common.CLASSES[d])
        bias = np.zeros(len(classes))
        best = macro_f1(labels, classes[(log_p + bias).argmax(1)])
        for _ in range(passes):
            for c in range(len(classes)):
                for value in grid:
                    trial = bias.copy()
                    trial[c] = value
                    f1 = macro_f1(labels, classes[(log_p + trial).argmax(1)])
                    if f1 > best + 1e-9:
                        best, bias = f1, trial
        biases[d] = bias
    return biases


def apply_biases(probabilities, biases):
    """Probabilities with the offsets applied (renormalized), so argmax gives the tuned answer."""
    result = {}
    for i, p in probabilities.items():
        result[i] = {}
        for d in common.DIMENSIONS:
            q = np.exp(np.log(p[d] + 1e-9) + biases[d])
            result[i][d] = q / q.sum()
    return result


def line(name, metrics):
    return (f"{name:28} macro F1 {metrics['overall']['f1_macro']:.3f} | weighted {metrics['overall']['f1_weighted']:.3f}"
            + "".join(f" | {d[:8]} {metrics[d]['f1_macro']:.2f}" for d in common.DIMENSIONS if d in metrics))


def write_submission(data_dir, answers, path):
    sample = pd.read_csv(data_dir / "sample_submission.csv", dtype=str, keep_default_na=False)
    submission = sample.set_index("image_id")
    predicted = [i for i in submission.index if i in answers]
    submission.loc[predicted, list(common.DIMENSIONS)] = [[answers[i][d] for d in common.DIMENSIONS] for i in predicted]
    for d in common.DIMENSIONS:
        assert submission[d].isin(common.CLASSES[d]).all(), f"{d} has an invalid cell"
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.reset_index().to_csv(path, index=False)
    print(f"{path}: {len(predicted)} dev rows predicted, final rows keep the constants")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--heads", action="append", default=[], help="NAME=FEATURES_DIR=LAYERS:POOLING:HIDDEN")
    parser.add_argument("--probs", action="append", default=[], help="NAME=HOLDOUT_JSONL=DEV_JSONL")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--step", type=float, default=0.25, help="grid step of the weight search")
    parser.add_argument("--tuned", action="store_true", help="dev submission with the weights tuned on the holdout")
    parser.add_argument("--bias", action="store_true", help="dev submission with class biases tuned on the holdout")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ensemble"))
    arguments = parser.parse_args()

    parts = load_parts(arguments.data_dir)
    metric = common.load_metric(arguments.data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sources = {}
    for spec in arguments.heads:
        name, rest = spec.split("=", 1)
        sources[name] = heads_probabilities(rest, parts, arguments.seeds, arguments.epochs, device)
        print(f"{name}: heads trained with {arguments.seeds} seeds", flush=True)
    for spec in arguments.probs:
        name, rest = spec.split("=", 1)
        sources[name] = jsonl_probabilities(rest)

    # Score on the holdout images every source has.
    common_ids = set.intersection(*(set(s["holdout"]) for s in sources.values()))
    rows = parts["holdout"][parts["holdout"]["image_id"].isin(common_ids)].reset_index(drop=True)
    holdout = {name: s["holdout"] for name, s in sources.items()}
    print(f"\nholdout: {len(rows)} images with context, scored like the leaderboard\n")
    for name in sources:
        print(line(name, score(metric, rows, holdout[name])))
    equal = {d: {name: 1.0 for name in sources} for d in common.DIMENSIONS}
    print(line("mean of all", score(metric, rows, combine(holdout, equal, rows["image_id"]))))
    for a, b in itertools.combinations(sources, 2):
        pair = {d: {a: 1.0, b: 1.0} for d in common.DIMENSIONS}
        print(line(f"mean {a} + {b}", score(metric, rows, combine({a: holdout[a], b: holdout[b]}, pair, rows["image_id"]))))

    tuned = search_weights(metric, rows, holdout, arguments.step)
    print(line("tuned on holdout (optimistic)", score(metric, rows, combine(holdout, tuned, rows["image_id"]))))
    print("  weights: " + " | ".join(f"{d[:8]} " + ",".join(f"{n}={w:.2f}" for n, w in tuned[d].items())
                                     for d in common.DIMENSIONS))

    # Honest estimate of the weight search: tune on half the holdout products, score on the other half.
    products = sorted(rows["product_id"].unique())
    random.Random(common.SEED).shuffle(products)
    halves = [set(products[: len(products) // 2]), set(products[len(products) // 2 :])]
    for label, combination in (("mean of all", None), ("tuned weights", "tuned")):
        scores = []
        for k in range(2):
            fit_rows = rows[rows["product_id"].isin(halves[k])].reset_index(drop=True)
            test_rows = rows[rows["product_id"].isin(halves[1 - k])].reset_index(drop=True)
            weights = search_weights(metric, fit_rows, holdout, arguments.step) if combination else equal
            scores.append(score(metric, test_rows, combine(holdout, weights, test_rows["image_id"]))["overall"]["f1_macro"])
        print(f"2-fold over holdout products, {label:14}: {np.mean(scores):.3f} ({', '.join(f'{s:.3f}' for s in scores)})")

    # Bias tuning on the mean: honest estimate with the same two halves, then biases from the whole holdout.
    mean_holdout = combine(holdout, equal, rows["image_id"])
    reference = score(metric, rows, mean_holdout)
    for d in common.DIMENSIONS:
        predicted = np.array([common.CLASSES[d][int(mean_holdout[i][d].argmax())] for i in rows["image_id"]])
        assert abs(macro_f1(scored_labels(rows, d), predicted) - reference[d]["f1_macro"]) < 1e-6, f"macro F1 of {d}"
    scores = []
    for k in range(2):
        fit_rows = rows[rows["product_id"].isin(halves[k])].reset_index(drop=True)
        test_rows = rows[rows["product_id"].isin(halves[1 - k])].reset_index(drop=True)
        biases = tune_biases(mean_holdout, fit_rows)
        test = {i: mean_holdout[i] for i in test_rows["image_id"]}
        scores.append(score(metric, test_rows, apply_biases(test, biases))["overall"]["f1_macro"])
    print(f"2-fold over holdout products, {'mean + biases':14}: {np.mean(scores):.3f} ({', '.join(f'{s:.3f}' for s in scores)})")
    biases = tune_biases(mean_holdout, rows)
    print(line("mean + biases (optimistic)", score(metric, rows, apply_biases(mean_holdout, biases))))
    print("  biases: " + " | ".join(f"{d[:8]} " + ",".join(f"{c}={b:+.2f}" for c, b in zip(common.CLASSES[d], biases[d]) if b)
                                    for d in common.DIMENSIONS))

    dev_ids = set.intersection(*(set(s["dev"]) for s in sources.values()))
    if dev_ids:
        weights = tuned if arguments.tuned else equal
        dev = combine({name: s["dev"] for name, s in sources.items()}, weights, sorted(dev_ids))
        name = "tuned" if arguments.tuned else "mean"
        if arguments.bias:
            dev, name = apply_biases(dev, biases), name + "_biases"
        write_submission(arguments.data_dir, argmax_answers(dev), arguments.output_dir / f"submission_ensemble_{name}.csv")
        np.save(arguments.output_dir / f"dev_probabilities_{name}.npy", {i: p for i, p in dev.items()}, allow_pickle=True)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(arguments.output_dir / "holdout_probabilities_mean.npy", mean_holdout, allow_pickle=True)
    np.save(arguments.output_dir / "biases.npy", biases, allow_pickle=True)


if __name__ == "__main__":
    main()
