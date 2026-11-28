"""Average the class probabilities of several models and write one submission file.

    python src/blend.py --data-dir ~/galaxus/data --output outputs/submission_blend.csv \
        --source lora=runs/lora_classifier/predictions \
        --source heads=outputs/oof_mirror/holdout_probabilities.npy,outputs/oof_mirror/dev_probabilities.npy

A source is a name and its prediction files: folders (every .jsonl in them) or single files, separated by commas.
A .jsonl file holds one line per image, {"image_id", "probabilities": {dimension: {class: p}}}, as
lora_classifier.py and lora_atomic.py write it. A .csv file holds one row per image and one column
"dimension:class" per probability. A .npy file holds image_id -> dimension -> probabilities in common.CLASSES
order, as oof.py and ensemble.py write it, or image_id -> the ten perspective probabilities, as dino_finetune.py
writes it.

A source may cover only some dimensions, such as a model for the perspective alone. Each dimension is averaged over
the sources that have it. A source can also be limited to some dimensions with @ (--source
lora@text_language+text_classification=...), and a weight can follow (--source lora:2=...).

The holdout images that every source has are scored with metric.py: each source alone (a source without all six
dimensions is filled up from the first source), then the mean. Every image of sample_submission.csv that all
sources predicted gets the class with the highest mean probability; the other rows keep what --base holds
(default: the constants of sample_submission.csv).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import common


def read_file(path):
    """image_id -> dimension -> probabilities of one prediction file."""
    if path.suffix == ".jsonl":
        result = {}
        for line in path.read_text().splitlines():
            record = json.loads(line)
            result[str(record["image_id"])] = {
                d: np.array([record["probabilities"][d][name] for name in common.CLASSES[d]], dtype=np.float64)
                for d in common.DIMENSIONS}
        return result
    if path.suffix == ".csv":
        table = pd.read_csv(path, dtype={"image_id": str}).set_index("image_id")
        covered = [d for d in common.DIMENSIONS if all(f"{d}:{name}" in table.columns for name in common.CLASSES[d])]
        values = {d: table[[f"{d}:{name}" for name in common.CLASSES[d]]].to_numpy(dtype=np.float64) for d in covered}
        return {image_id: {d: values[d][k] for d in covered} for k, image_id in enumerate(table.index)}
    loaded = np.load(path, allow_pickle=True).item()
    result = {}
    for image_id, p in loaded.items():
        if isinstance(p, dict):
            result[str(image_id)] = {d: np.asarray(p[d], dtype=np.float64) for d in common.DIMENSIONS if d in p}
        elif len(p) == len(common.CLASSES["perspective"]):  # dino_finetune.py: the perspective probabilities alone
            result[str(image_id)] = {"perspective": np.asarray(p, dtype=np.float64)}
    return result


def read_source(spec):
    """(name, weight, image_id -> dimension -> probabilities) of a --source argument."""
    label, files = spec.split("=", 1)
    label, _, weight = label.partition(":")
    name, _, only = label.partition("@")
    kept = [d for d in common.DIMENSIONS if not only or d in only.split("+")]
    assert not only or set(only.split("+")) <= set(common.DIMENSIONS), f"unknown dimension in {label}"
    probabilities = {}
    for entry in files.split(","):
        path = Path(entry).expanduser()
        paths = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        for file in paths:
            probabilities.update(read_file(file))
    probabilities = {image_id: {d: p[d] for d in kept if d in p} for image_id, p in probabilities.items()}
    return label, float(weight or 1), probabilities


def mean_of(sources, image_id):
    """dimension -> weighted mean of the probabilities of the sources that have this dimension, for one image."""
    result = {}
    for d in common.DIMENSIONS:
        having = [(weight, p[image_id][d]) for _, weight, p in sources if d in p[image_id]]
        result[d] = sum(weight * values for weight, values in having) / sum(weight for weight, _ in having)
    return result


def answers_of(probabilities):
    return {image_id: {d: common.CLASSES[d][int(np.argmax(p[d]))] for d in common.DIMENSIONS}
            for image_id, p in probabilities.items()}


def line(name, metrics):
    return (f"{name:28} {metrics['overall']['f1_macro']:.4f} | weighted {metrics['overall']['f1_weighted']:.4f} | "
            + " | ".join(f"{d[:8]} {metrics[d]['f1_macro']:.3f}" for d in common.DIMENSIONS if d in metrics))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True, help="name[:weight]=file_or_folder[,more]")
    parser.add_argument("--output", type=Path, required=True, help="the submission file to write")
    parser.add_argument("--base", type=Path, help="a submission whose rows are kept where the sources predict nothing")
    arguments = parser.parse_args()

    sources = [read_source(spec) for spec in arguments.source]
    assert all(d in sources[0][2][next(iter(sources[0][2]))] for d in common.DIMENSIONS), "the first source needs all six dimensions"
    for name, weight, probabilities in sources:
        print(f"{name}: {len(probabilities)} images, weight {weight:g}")
    shared = set.intersection(*(set(p) for _, _, p in sources))

    _, holdout = common.split_holdout(common.read_part(arguments.data_dir / "train.csv"))
    rows = holdout[holdout["image_id"].isin(shared)].reset_index(drop=True)
    if len(rows):
        metric = common.load_metric(arguments.data_dir)
        print(f"\nholdout: {len(rows)} of {len(holdout)} images are in every source\n")
        for source in sources:
            alone = {image_id: {**sources[0][2][image_id], **source[2][image_id]} for image_id in rows["image_id"]}
            print(line(source[0], common.score_rows(metric, rows, answers_of(alone), {})))
        if len(sources) > 1:
            blended = {image_id: mean_of(sources, image_id) for image_id in rows["image_id"]}
            print(line("mean", common.score_rows(metric, rows, answers_of(blended), {})))
    else:
        print("\nno holdout image is in every source, so nothing is scored")

    base = arguments.base or arguments.data_dir / "sample_submission.csv"
    submission = pd.read_csv(base, dtype=str, keep_default_na=False).set_index("image_id")
    predicted = [image_id for image_id in submission.index if image_id in shared]
    answers = answers_of({image_id: mean_of(sources, image_id) for image_id in predicted})
    if predicted:
        submission.loc[predicted, list(common.DIMENSIONS)] = [[answers[i][d] for d in common.DIMENSIONS] for i in predicted]
    for d in common.DIMENSIONS:
        assert submission[d].isin(common.CLASSES[d]).all(), f"{d} has an invalid cell"
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    submission.reset_index().to_csv(arguments.output, index=False)
    print(f"\n{arguments.output}: {len(predicted)} of {len(submission)} rows from the sources, the rest from {base.name}")


if __name__ == "__main__":
    main()
