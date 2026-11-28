"""text_language from OCR where the image holds enough readable text: the OCR language overrides the heads.

EasyOCR reads the text boxes (saved by ocr_features.py as ocr.jsonl), Lingua names the language of the joined text.
Where at least MIN_LETTERS letters are read with confidence >= MIN_READ and Lingua is at least MIN_LANGUAGE sure,
text_language becomes that language (German, English, French, Italian, Dutch, else other). Elsewhere the heads decide.
Tuned on out-of-fold train predictions: text_language macro F1 0.616 -> 0.656 (11k images), holdout 0.711 -> 0.718.

    python src/ocr_rule.py --probabilities outputs/oof_mirror/dev_probabilities.npy --ocr results/ocr/dev \
        --data-dir ~/galaxus/data --output outputs/submission_dev.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np

import common

MIN_READ, MIN_LETTERS, MIN_LANGUAGE = 0.3, 10, 0.5


def read_ocr(folder):
    """image_id -> OCR boxes, from every ocr*.jsonl in the folder."""
    boxes = {}
    for path in sorted(Path(folder).glob("ocr*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            boxes[record["image_id"]] = record["boxes"]
    return boxes


def detector():
    from lingua import Language, LanguageDetectorBuilder

    names = {Language.GERMAN: "German", Language.ENGLISH: "English", Language.FRENCH: "French",
             Language.ITALIAN: "Italian", Language.DUTCH: "Dutch"}
    return LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build(), names


def ocr_language(boxes, lingua, names):
    """The language class the OCR text votes for, or None when there is too little text or Lingua is unsure."""
    text = " ".join(box["text"] for box in boxes if box["conf"] >= MIN_READ)
    if sum(character.isalpha() for character in text) < MIN_LETTERS:
        return None
    values = lingua.compute_language_confidence_values(text)
    if not values or values[0].value < MIN_LANGUAGE:
        return None
    return names.get(values[0].language, "other")


def apply(answers, ocr):
    """answers: image_id -> dimension -> class. Returns a copy with text_language overridden where the OCR is clear."""
    lingua, names = detector()
    result, changed = {}, 0
    for image_id, labels in answers.items():
        result[image_id] = dict(labels)
        language = ocr_language(ocr.get(image_id, []), lingua, names)
        if language and language != labels["text_language"]:
            result[image_id]["text_language"] = language
            changed += 1
    print(f"OCR rule: text_language changed on {changed} of {len(answers)} images", flush=True)
    return result


def main():
    from ensemble import argmax_answers, write_submission

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probabilities", type=Path, required=True, help="image_id -> dimension -> probabilities (.npy)")
    parser.add_argument("--ocr", type=Path, required=True, help="folder with the ocr*.jsonl of these images")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    probabilities = np.load(arguments.probabilities, allow_pickle=True).item()
    answers = apply(argmax_answers(probabilities), read_ocr(arguments.ocr))
    assert all(a["text_language"] in common.CLASSES["text_language"] for a in answers.values())
    write_submission(arguments.data_dir, answers, arguments.output)


if __name__ == "__main__":
    main()
