"""Score the question tree with answers from a local Apertus (inference only, nothing trained).

tree_features.py stores P(yes) for the tree's yes/no questions and the option probabilities of the six atomic
multiple-choice questions. Here they become answers (yes when P(yes) >= 0.5, the most likely option otherwise) and the
unchanged decision tree of the question tree (src/question_tree/tree.py, with its kept rules) turns them into labels.
White or not comes from the pixel measurement, as in the question tree. Perspective comes from the multiple-choice
answer (the tree's fallback), since the contour model of the question tree would have to be trained.

    python src/tree_inference.py --data-dir ~/galaxus/data --answers results/tree_4bit_base --pixels results/pixels
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np

import common
from atomic import QUESTIONS as ATOMIC
from heads import load_features, load_parts

TREE_FILE = Path(__file__).parent / "question_tree" / "tree.py"
WHITE_INDEX = 16  # pixel_features.backdrop_features: 16 corner medians, then the white flag
CHOICE_KEYS = {"subject": "subject_mc", "text_classification": "overlay_mc", "text_language": "lang",
               "perspective": "persp_mc", "background": "bg_mc"}


def load_tree():
    spec = importlib.util.spec_from_file_location("tree_rules", TREE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def answers_of(names, values):
    """{question key: answer} of one image, from its stored probabilities."""
    answers, position = {}, 0
    yes_no = [n for n in names if "=" not in n]
    for key, p in zip(yes_no, values):
        answers[key] = "yes" if p >= 0.5 else "no"
    position = len(yes_no)
    for d in common.DIMENSIONS:
        options = [name for name, _ in ATOMIC[d][1]]
        probabilities = values[position : position + len(options)]
        position += len(options)
        if d in CHOICE_KEYS:
            answers[CHOICE_KEYS[d]] = options[int(np.argmax(probabilities))]
    return answers


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True, help="features folder of tree_features.py")
    parser.add_argument("--pixels", type=Path, default=Path("results/pixels"))
    arguments = parser.parse_args()

    tree = load_tree()
    metric = common.load_metric(arguments.data_dir)
    holdout = load_parts(arguments.data_dir)["holdout"]
    stored = np.load(sorted((arguments.answers / "holdout").glob("shard_*.npz"))[0])
    names, ids, values = list(stored["names"]), stored["image_ids"], stored["features"][:, 0, 0]
    pixel_ids, _, _, pixel_features, _ = load_features(arguments.pixels, "holdout")
    white = {i: bool(f[0, 0, WHITE_INDEX] >= 0.5) for i, f in zip(pixel_ids, pixel_features)}

    rows = holdout[holdout["image_id"].isin(set(ids))].reset_index(drop=True)
    by_id = {i: answers_of(names, v) for i, v in zip(ids, values)}
    results = {
        "tree with kept rules": {i: tree.classify(a, white[i]) for i, a in by_id.items()},
        "tree without the sharpened rules": {i: tree.classify(a, white[i], tree.NO_RULES) for i, a in by_id.items()},
        "one multiple-choice question per dimension": {i: tree.multiple_choice_only(a) for i, a in by_id.items()},
    }
    print(f"{len(rows)} holdout images, answers from {arguments.answers}\n")
    for name, answers in results.items():
        fixed = {i: {d: a.get(d) if a.get(d) in common.CLASSES[d] else "none" if "none" in common.CLASSES[d]
                     else common.CLASSES[d][0] for d in common.DIMENSIONS} for i, a in answers.items()}
        metrics = common.score_rows(metric, rows, fixed, {})
        print(f"{name:45} macro F1 {metrics['overall']['f1_macro']:.3f} | weighted {metrics['overall']['f1_weighted']:.3f}"
              + "".join(f" | {d[:8]} {metrics[d]['f1_macro']:.2f}" for d in common.DIMENSIONS if d in metrics))


if __name__ == "__main__":
    main()
