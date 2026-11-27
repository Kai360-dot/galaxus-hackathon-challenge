"""Scoring of the Hack Apertus image type challenge, used as the Kaggle metric and for local evaluation.

A submission has one row per image and one column per dimension holding a class name. The solution holds
the human label of every scored cell and is empty where a cell is not scored: the label is only a model
prelabel, or the relevance rules make the dimension meaningless for that image. Empty submission cells
count as wrong answers.

The leaderboard number is the overall macro F1: the mean over the six dimensions of each dimension's
macro F1. The overall weighted F1 pools all scored cells, so it is the support weighted mean of the
dimension weighted F1 scores.
"""

import math
from typing import Any

import pandas as pd

DIMENSIONS = ("subject", "background", "perspective", "individuals", "text_classification", "text_language")

VALID_CLASSES = {
    "subject": frozenset(
        {
            "full_product",
            "detail_closeup",
            "accessories",
            "packaging",
            "package_and_product",
            "multipack_trade_unit",
            "delivery_scope",
            "product_grouping",
            "multiview",
            "energy_label",
            "infographic",
            "logo",
            "placeholder",
        }
    ),
    "background": frozenset({"white", "color", "scene_lifestyle"}),
    "perspective": frozenset(
        {
            "front_head_on",
            "front_angle_left",
            "front_angle_right",
            "back_head_on",
            "back_angle_left",
            "back_angle_right",
            "left_side",
            "right_side",
            "top_down",
            "bottom_up",
        }
    ),
    "individuals": frozenset({"individual_present", "none"}),
    "text_classification": frozenset(
        {"certification_seal", "technical_specs", "promotional_text", "other", "nutritional_info", "none"}
    ),
    "text_language": frozenset({"German", "English", "French", "Italian", "Dutch", "other", "none"}),
}

USAGE_COLUMN = "Usage"
EXAMPLE_COUNT = 5


class ParticipantVisibleError(Exception):
    """An error whose message Kaggle shows to the participant."""


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """The leaderboard number: the mean over the dimensions of the macro F1."""
    return evaluate(solution, submission, row_id_column_name)["overall"]["f1_macro"]


def evaluate(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> dict[str, Any]:
    """F1, precision and recall per class, per dimension and overall."""
    joined = _join_predictions(solution, submission, row_id_column_name)
    metrics: dict[str, Any] = {}
    for dimension in DIMENSIONS:
        scored = joined[joined[f"{dimension}_label"].notna()]
        if scored.empty:
            continue
        pairs = list(zip(scored[f"{dimension}_label"], scored[f"{dimension}_prediction"], strict=True))
        metrics[dimension] = _dimension_metrics(pairs, VALID_CLASSES[dimension])
    if not metrics:
        raise ValueError("The solution has no scored cell.")
    total_support = sum(dimension_metrics["support"] for dimension_metrics in metrics.values())
    metrics["overall"] = {
        "f1_macro": sum(dimension_metrics["f1_macro"] for dimension_metrics in metrics.values()) / len(metrics),
        "f1_weighted": sum(
            dimension_metrics["f1_weighted"] * dimension_metrics["support"] for dimension_metrics in metrics.values()
        )
        / total_support,
        "support": total_support,
    }
    return metrics


def _join_predictions(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> pd.DataFrame:
    _check_columns(submission, row_id_column_name)
    _check_row_ids(solution, submission, row_id_column_name)
    labels = solution.drop(columns=[USAGE_COLUMN], errors="ignore")[[row_id_column_name, *DIMENSIONS]]
    labels = labels.set_index(row_id_column_name).map(_cell_value)
    predictions = submission[[row_id_column_name, *DIMENSIONS]].set_index(row_id_column_name).map(_cell_value)
    _check_classes(predictions)
    return labels.join(predictions, lsuffix="_label", rsuffix="_prediction", how="left")


def _check_columns(submission: pd.DataFrame, row_id_column_name: str) -> None:
    missing = [column for column in (row_id_column_name, *DIMENSIONS) if column not in submission.columns]
    if missing:
        raise ParticipantVisibleError(f"The submission misses the columns {missing}.")


def _check_row_ids(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> None:
    if submission[row_id_column_name].duplicated().any():
        duplicates = submission.loc[submission[row_id_column_name].duplicated(), row_id_column_name]
        raise ParticipantVisibleError(f"Duplicated ids in the submission, for example {list(duplicates[:EXAMPLE_COUNT])}.")
    missing = set(solution[row_id_column_name]) - set(submission[row_id_column_name])
    if missing:
        raise ParticipantVisibleError(f"The submission misses {len(missing)} ids, for example {sorted(missing)[:EXAMPLE_COUNT]}.")


def _check_classes(predictions: pd.DataFrame) -> None:
    for dimension in DIMENSIONS:
        values = predictions[dimension].dropna()
        invalid = sorted(set(values) - VALID_CLASSES[dimension])
        if invalid:
            more = f" and {len(invalid) - 1} more" if len(invalid) > 1 else ""
            raise ParticipantVisibleError(
                f"{dimension}: unknown class {invalid[0]!r}{more}. Allowed: {', '.join(sorted(VALID_CLASSES[dimension]))}."
            )


def _cell_value(value: object) -> str | None:
    """The class name of a cell, or None for an empty cell."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _dimension_metrics(pairs: list[tuple[str, str | None]], valid_classes: frozenset[str]) -> dict[str, Any]:
    # The classes follow scikit-learn's convention: every class of the labels plus every valid class that was
    # predicted, so predicting a class absent from the labels adds a class with F1 0 to the macro mean. An
    # empty prediction is wrong for its label but never becomes a class of its own.
    label_counts: dict[str, int] = {}
    prediction_counts: dict[str, int] = {}
    hits: dict[str, int] = {}
    for label, prediction in pairs:
        label_counts[label] = label_counts.get(label, 0) + 1
        if prediction is not None:
            prediction_counts[prediction] = prediction_counts.get(prediction, 0) + 1
        if prediction == label:
            hits[label] = hits.get(label, 0) + 1
    classes = sorted(set(label_counts) | (set(prediction_counts) & valid_classes))
    per_class = {
        class_name: _class_metrics(
            hits.get(class_name, 0), prediction_counts.get(class_name, 0), label_counts.get(class_name, 0)
        )
        for class_name in classes
    }
    support = len(pairs)
    return {
        "f1_macro": sum(metrics["f1"] for metrics in per_class.values()) / len(per_class),
        "f1_weighted": sum(metrics["f1"] * metrics["support"] for metrics in per_class.values()) / support,
        "precision_macro": sum(metrics["precision"] for metrics in per_class.values()) / len(per_class),
        "recall_macro": sum(metrics["recall"] for metrics in per_class.values()) / len(per_class),
        "support": support,
        "per_class": per_class,
    }


def _class_metrics(hit_count: int, prediction_count: int, label_count: int) -> dict[str, float | int]:
    precision = hit_count / prediction_count if prediction_count else 0.0
    recall = hit_count / label_count if label_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": label_count}
