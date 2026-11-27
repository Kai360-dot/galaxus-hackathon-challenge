"""The whole pipeline for a csv of images.

    cd src
    python -m question_tree.predict --csv ../../glx_data/dev.csv --name dev

Per image: Apertus answers the questions of the tree, the backdrop is measured in pixels, the tree sets five
dimensions and the contour model sets the perspective. The predictions are written next to the stored answers,
outside the repository. With --submission they are also merged into a copy of sample_submission.csv.
Stored answers are reused, so an interrupted run continues where it stopped.
"""

import argparse
from pathlib import Path

import pandas as pd

from . import ask, contours, data, evaluate
from .config import DATA_DIR, DIMENSIONS, GROUP_ROBUST_QUESTIONS, MODEL_8B, STORE, WORKERS
from .questions import TREE_QUESTIONS


def predict(rows, name, model=MODEL_8B, group=GROUP_ROBUST_QUESTIONS, workers=WORKERS):
    """The six labels for every image of rows. rows needs image_id, image_path and the product context columns."""
    answers_path = STORE / f"answers_{name}_{model}.jsonl"
    speed = ask.collect(model, rows, TREE_QUESTIONS, answers_path, group=group, workers=workers)
    answers = ask.read_answers(answers_path).reindex(rows["image_id"])
    predictions = evaluate.tree_predictions(answers, evaluate.measure_backdrops(rows))

    train = data.load_train()
    labeled = train[data.is_scored(train, "perspective")]
    model_of_view = contours.PerspectiveModel().fit(labeled, contours.measure(labeled))
    predictions["perspective"] = model_of_view.predict(rows, contours.measure(rows))
    print(f"{len(rows)} images, {speed:.2f} images/s for the questions")
    return predictions[["image_id", *DIMENSIONS]]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, required=True, help="csv with image_id, image_path and the product context")
    parser.add_argument("--name", required=True, help="short name of this image set, used in the file names")
    parser.add_argument("--model", default=MODEL_8B)
    parser.add_argument("--limit", type=int, help="only the first N images")
    parser.add_argument("--workers", type=int, default=WORKERS, help="images handled in parallel")
    parser.add_argument("--no-grouping", action="store_true", help="ask every question alone")
    parser.add_argument("--submission", action="store_true", help="also write a full submission file")
    arguments = parser.parse_args()

    csv_path = arguments.csv.resolve()
    rows = pd.read_csv(csv_path)
    rows["image_path"] = [str((csv_path.parent / path).relative_to(DATA_DIR)) for path in rows["image_path"]]
    if arguments.limit:
        rows = rows.head(arguments.limit)
    group = GROUP_ROBUST_QUESTIONS and not arguments.no_grouping
    predictions = predict(rows, arguments.name, arguments.model, group=group, workers=arguments.workers)
    out_path = STORE / f"predictions_{arguments.name}_{arguments.model}.csv"
    predictions.to_csv(out_path, index=False)
    print(f"predictions in {out_path}")
    if arguments.submission:
        submission = pd.read_csv(DATA_DIR / "sample_submission.csv", keep_default_na=False).set_index("image_id")
        submission.loc[predictions["image_id"], list(DIMENSIONS)] = predictions[list(DIMENSIONS)].to_numpy()
        submission_path = STORE / f"submission_{arguments.name}_{arguments.model}.csv"
        submission.reset_index().to_csv(submission_path, index=False)
        print(f"submission with {len(predictions)} predicted rows in {submission_path}")


if __name__ == "__main__":
    main()
