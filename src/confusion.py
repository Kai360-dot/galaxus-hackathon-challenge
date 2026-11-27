"""Confusion matrices on the holdout, per model and dimension, as JSON for a report page.

Every model is a file of holdout probabilities (ensemble.py --save-sources), or a mean of several, or a file of
labels ({image_id: {dimension: class}}, for example from tree_inference.py --save). Only scored cells count, as on
the leaderboard: human labels in relevant cells.

    python src/confusion.py --data-dir ~/galaxus/data --sources outputs/confusion_sources \
        --model "Heads, Apertus only=apertus_only" --model "Best: mean of 3 layers=A16+A24+A32" \
        --output outputs/confusion.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

import common
from heads import load_parts


def load_probabilities(sources_dir, name):
    return np.load(sources_dir / f"holdout_probabilities_{name}.npy", allow_pickle=True).item()


def labels_of(spec, sources_dir):
    """{image_id: {dimension: class}} of a model spec: name, name+name (mean), or labels:path.json."""
    if spec.startswith("labels:"):
        return json.loads(Path(spec[len("labels:"):]).read_text())
    parts = [load_probabilities(sources_dir, name) for name in spec.split("+")]
    ids = set.intersection(*(set(p) for p in parts))
    return {i: {d: common.CLASSES[d][int(np.mean([p[i][d] for p in parts], axis=0).argmax())]
                for d in common.DIMENSIONS} for i in ids}


def confusion(rows, labels, d):
    scored = (rows[f"{d}_source"] == "human") & (rows[f"{d}_relevant"].str.lower() == "true")
    pairs = [(row[d], labels[row["image_id"]][d]) for _, row in rows[scored].iterrows() if row["image_id"] in labels]
    classes = list(common.CLASSES[d])
    index = {c: k for k, c in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for true, predicted in pairs:
        matrix[index[true], index[predicted]] += 1
    present = sorted({index[t] for t, _ in pairs} | {index[p] for _, p in pairs})  # as metric.py: labels + predictions
    per_class = {}
    for k in present:
        tp, fp, fn = matrix[k, k], matrix[:, k].sum() - matrix[k, k], matrix[k].sum() - matrix[k, k]
        per_class[classes[k]] = {
            "support": int(matrix[k].sum()), "predicted": int(matrix[:, k].sum()),
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0, "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else 0.0,
        }
    return {"classes": classes, "matrix": matrix.tolist(), "per_class": per_class,
            "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])), "cells": len(pairs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sources", type=Path, default=Path("outputs/confusion_sources"))
    parser.add_argument("--model", action="append", required=True, help="TITLE=SPEC (name, a+b+c or labels:file)")
    parser.add_argument("--output", type=Path, default=Path("outputs/confusion.json"))
    arguments = parser.parse_args()
    holdout = load_parts(arguments.data_dir)["holdout"]
    result = {}
    for model in arguments.model:
        title, spec = model.split("=", 1)
        labels = labels_of(spec, arguments.sources)
        result[title] = {d: confusion(holdout, labels, d) for d in common.DIMENSIONS}
        overall = np.mean([result[title][d]["macro_f1"] for d in common.DIMENSIONS])
        print(f"{title:50} macro F1 {overall:.3f} | "
              + " | ".join(f"{d[:8]} {result[title][d]['macro_f1']:.2f}" for d in common.DIMENSIONS))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result))
    print(f"confusion matrices in {arguments.output}")


if __name__ == "__main__":
    main()
