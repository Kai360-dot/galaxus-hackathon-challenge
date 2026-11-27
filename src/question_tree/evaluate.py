"""Turning answers into predictions and scoring them with the leaderboard metric."""

import json

import pandas as pd

from . import pixels, tree
from .config import CONSTANT_ANSWER, DATA_DIR, DIMENSIONS, STORE, metric
from .data import solution_frame
from .images import load_rgb


def measure_backdrops(rows):
    """Per image: is the backdrop white (with and without the frame tolerance) and the four corner medians."""
    cache_path = STORE / "backdrops.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for row in rows.itertuples():
        if row.image_id not in cache:
            image = load_rgb(DATA_DIR / row.image_path)
            cache[row.image_id] = {
                "white": bool(pixels.white_backdrop(image)),
                "white_at_border": bool(pixels.white_backdrop(image, tolerate_frame=False)),
                "corners": pixels.corner_medians(pixels.brightness(image)),
            }
    cache_path.write_text(json.dumps(cache))
    return pd.DataFrame({image_id: cache[image_id] for image_id in rows["image_id"]}).T


def tree_predictions(answers, backdrops, rules=tree.KEPT_RULES, white_column="white"):
    """The tree applied to every image of the answer table."""
    records = [
        {"image_id": image_id, **tree.classify(row.dropna().to_dict(), bool(backdrops.loc[image_id, white_column]), rules)}
        for image_id, row in answers.iterrows()
    ]
    return pd.DataFrame(records)


def multiple_choice_predictions(answers):
    return pd.DataFrame([{"image_id": image_id, **tree.multiple_choice_only(row.dropna().to_dict())} for image_id, row in answers.iterrows()])


def constant_predictions(rows):
    return rows[["image_id"]].assign(**CONSTANT_ANSWER)


def with_perspective(predictions, perspective):
    """A copy of predictions whose perspective comes from the series perspective (indexed by image_id) where it has one."""
    result = predictions.copy()
    result["perspective"] = [perspective.get(image_id, current) for image_id, current in zip(result["image_id"], result["perspective"])]
    return result


def evaluate(predictions, rows):
    """The metric's full result for one prediction frame, on the images of rows that it covers."""
    covered = rows[rows["image_id"].isin(set(predictions["image_id"]))]
    frame = predictions[predictions["image_id"].isin(set(covered["image_id"]))][["image_id", *DIMENSIONS]].copy()
    for dimension in DIMENSIONS:  # an answer that is not a class name counts as empty, which the metric scores as wrong
        frame[dimension] = frame[dimension].where(frame[dimension].isin(list(metric.VALID_CLASSES[dimension])))
    return metric.evaluate(solution_frame(covered), frame, "image_id")


def score_table(methods, rows):
    """Macro F1 per dimension, the leaderboard score and the weighted F1 for every method, on the images all of them cover."""
    common = set(rows["image_id"]).intersection(*(set(frame["image_id"]) for frame in methods.values()))
    shared = rows[rows["image_id"].isin(common)]
    table = {}
    for name, frame in methods.items():
        result = evaluate(frame, shared)
        table[name] = {**{d: result[d]["f1_macro"] for d in DIMENSIONS if d in result}, "overall": result["overall"]["f1_macro"], "weighted": result["overall"]["f1_weighted"]}
    table = pd.DataFrame(table).T.round(3)
    table.insert(0, "images", len(shared))
    return table


def read_strategy(path):
    """The stored labels of a one-prompt strategy as a prediction frame."""
    records = [json.loads(line) for line in path.read_text().splitlines()]
    return pd.DataFrame([{"image_id": record["image_id"], **record["labels"]} for record in records])


def read_baseline(model, runs_dir):
    """The answers of the one-prompt baseline, written by the zero-shot baseline notebook."""
    records = [json.loads(line) for line in (runs_dir / "zero_shot_v1" / f"{model}.jsonl").read_text().splitlines()]
    frame = pd.DataFrame([{"image_id": record["image_id"], **json.loads(record["answer"])} for record in records])
    return frame.drop_duplicates("image_id", keep="last")
