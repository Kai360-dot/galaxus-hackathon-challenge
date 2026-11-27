"""The train table, the image sets of the experiments and the frame the metric scores against."""

import pandas as pd

from .config import CLASSES, DATA_DIR, DIMENSIONS, RUNS_DIR, SEED, STORE


def load_train():
    return pd.read_csv(DATA_DIR / "train.csv")


def stored_ids(path, make):
    """The ids in path, one per line. The first call creates the file with make()."""
    if not path.exists():
        path.write_text("\n".join(make()) + "\n")
    return path.read_text().split()


def is_scored(rows, dimension):
    """True where the leaderboard would score the cell: a human label in a relevant cell."""
    return (rows[f"{dimension}_source"] == "human") & rows[f"{dimension}_relevant"]


def solution_frame(rows):
    """The labels that count, every other cell empty."""
    solution = rows[["image_id", *DIMENSIONS]].copy()
    for dimension in DIMENSIONS:
        solution[dimension] = solution[dimension].where(is_scored(rows, dimension))
    return solution


def holdout_mask(train):
    """True for the images of the holdout products: 20% of the products, never used to design or train anything."""
    products = stored_ids(
        RUNS_DIR / "holdout_products.txt",
        lambda: sorted(train["product_id"].drop_duplicates().sample(frac=0.2, random_state=SEED)),
    )
    return train["product_id"].isin(products)


def image_sets(train):
    """The three image sets of the experiments. No image is in two of them.

    tune       class-balanced images outside the holdout. Every question and every rule was designed here.
               It shares no product with the other two sets.
    test       300 random holdout images, unseen until the tree was frozen.
    benchmark  the 500 random holdout images of the zero-shot baseline notebook, also unseen.
    """
    holdout = train[holdout_mask(train)]
    outside = train[~holdout_mask(train)]

    def pick(dimension, count, seed):
        parts = []
        for name in CLASSES[dimension]:
            candidates = outside[(outside[dimension] == name) & is_scored(outside, dimension)]
            parts.append(candidates.sample(min(count, len(candidates)), random_state=seed))
        return pd.concat(parts)

    def tune_ids():
        people = outside[(outside["individuals"] == "individual_present") & is_scored(outside, "individuals")]
        overlays = pick("text_classification", 8, 1)
        parts = [
            pick("subject", 12, 1), people.sample(12, random_state=1), overlays[overlays["text_classification"] != "none"],
            pick("perspective", 12, 1), pick("background", 25, 2), pick("text_language", 12, 3),
        ]
        return list(pd.concat(parts)["image_id"].drop_duplicates())

    benchmark_ids = stored_ids(RUNS_DIR / "sample_500.txt", lambda: list(holdout["image_id"].sample(n=500, random_state=SEED)))
    fresh = holdout[~holdout["image_id"].isin(benchmark_ids)]
    ids = {
        "tune": stored_ids(STORE / "set_tune.txt", tune_ids),
        "test": stored_ids(STORE / "set_test.txt", lambda: list(fresh["image_id"].sample(300, random_state=7))),
        "benchmark": benchmark_ids,
    }
    by_id = train.set_index("image_id")
    return {name: by_id.loc[image_ids].reset_index() for name, image_ids in ids.items()}


def reference_holdout(train):
    """The 600-image holdout of the reference notebook, which the trained methods in this repository use.

    It comes from common.split_holdout, so the question tree is scored on exactly the images that the
    classification heads and the LoRA finetune are scored on.
    """
    import common  # the shared module of the local pipeline, next to this package

    _, holdout = common.split_holdout(common.read_part(DATA_DIR / "train.csv"))
    return train[train["image_id"].isin(set(holdout["image_id"]))].reset_index(drop=True)
