"""OCR features as helpers for the heads: what text the image holds, in which language, and where.

An open OCR model (EasyOCR, Apache-2.0) reads the text and a language detector (Lingua, Apache-2.0) names its
language. Both are supporting models, like DINOv2: Apertus and the heads still decide. Per image, 36 numbers:

- amount: boxes, characters, words, letters, share of the image area covered by text, mean confidence
- language: share of the text area per language (German, English, French, Italian, Dutch, other, unknown), and the
  same per word count
- position: share of the text area near the border (overlays sit there) and in the centre, top and bottom bands,
  height of the largest box
- keywords: nutrition terms, measurements, promotion words, percent signs, certification words

The raw OCR result of each image is kept beside the features (ocr.jsonl), so the features can be changed without
reading the images again. Stored in the shard layout of the Apertus features; mirrored train copies (only images
without text) get the features of the original.

    python src/ocr_features.py --data-dir ~/galaxus/data --features-dir results/ocr --parts holdout train dev
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import common
from heads import extraction_jobs, load_parts

LANGUAGES = ["German", "English", "French", "Italian", "Dutch"]
NUTRITION = re.compile(r"\b(kcal|kj|energie|energy|fett|fat|matières grasses|grassi|eiweiss|eiwei|protein|protéines|"
                       r"proteine|kohlenhydrate|carbohydrate|glucides|carboidrati|zucker|sugar|salz|salt|nährwert|"
                       r"nutrition|valeurs nutritionnelles|valori nutrizionali|ballaststoffe|fibre|rohprotein|rohasche)\b",
                       re.IGNORECASE)
MEASURE = re.compile(r"\d+([.,]\d+)?\s?(mm|cm|m|kg|g|ml|l|w|v|mah|gb|tb|hz|\"|''|inch|zoll)\b", re.IGNORECASE)
PROMOTION = re.compile(r"\b(neu|new|nouveau|nuovo|sale|aktion|gratis|free|bestseller|top|premium|jetzt|now|"
                       r"angebot|offer|exklusiv|limited|edition)\b|!", re.IGNORECASE)
SEAL = re.compile(r"\b(testsieger|test|winner|award|bio|organic|vegan|fsc|ce|tüv|gs|ecolabel|certified|zertifiziert|"
                  r"geprüft|approved|stiftung warentest|label)\b", re.IGNORECASE)
FEATURE_COUNT = 36


def language_detector():
    from lingua import Language, LanguageDetectorBuilder

    languages = [Language.GERMAN, Language.ENGLISH, Language.FRENCH, Language.ITALIAN, Language.DUTCH,
                 Language.SPANISH, Language.PORTUGUESE, Language.POLISH, Language.CZECH, Language.SWEDISH,
                 Language.DANISH, Language.TURKISH]
    return LanguageDetectorBuilder.from_languages(*languages).with_minimum_relative_distance(0.1).build()


def language_of(detector, text):
    """One of LANGUAGES, "other", or "unknown" when the text is too short or has no clear language."""
    letters = re.sub(r"[^A-Za-zÀ-ÿ ]", "", text).strip()
    if len(letters) < 4:
        return "unknown"
    language = detector.detect_language_of(letters)
    if language is None:
        return "unknown"
    name = language.name.capitalize()
    return name if name in LANGUAGES else "other"


def features_of(result, size, detector):
    """36 numbers from the OCR boxes of one image (see the module text)."""
    width, height = size
    area = float(width * height)
    boxes = []
    for points, text, confidence in result:
        xs, ys = [float(p[0]) for p in points], [float(p[1]) for p in points]
        boxes.append({"x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys), "text": text, "conf": float(confidence)})
    text = " ".join(b["text"] for b in boxes)
    words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", text)
    box_area = [max(0.0, (b["x1"] - b["x0"]) * (b["y1"] - b["y0"])) for b in boxes]
    total_area = sum(box_area)
    amount = [len(boxes), len(text) / 100, len(words) / 20, len(re.findall(r"[A-Za-zÀ-ÿ]", text)) / 100,
              min(1.0, total_area / area), float(np.mean([b["conf"] for b in boxes])) if boxes else 0.0]
    by_area = dict.fromkeys(LANGUAGES + ["other", "unknown"], 0.0)
    by_words = dict.fromkeys(LANGUAGES + ["other", "unknown"], 0.0)
    for b, a in zip(boxes, box_area):
        language = language_of(detector, b["text"])
        by_area[language] += a
        by_words[language] += len(re.findall(r"[A-Za-zÀ-ÿ]{2,}", b["text"]))
    whole = language_of(detector, text)
    languages = ([by_area[k] / total_area if total_area else 0.0 for k in by_area]
                 + [by_words[k] / max(1, len(words)) for k in by_words])
    border, centre, top, bottom, largest = 0.0, 0.0, 0.0, 0.0, 0.0
    for b, a in zip(boxes, box_area):
        cx, cy = (b["x0"] + b["x1"]) / 2 / width, (b["y0"] + b["y1"]) / 2 / height
        near_edge = min(cx, 1 - cx, cy, 1 - cy) < 0.15
        border += a if near_edge else 0.0
        centre += a if not near_edge else 0.0
        top += a if cy < 0.2 else 0.0
        bottom += a if cy > 0.8 else 0.0
        largest = max(largest, (b["y1"] - b["y0"]) / height)
    position = [v / total_area if total_area else 0.0 for v in (border, centre, top, bottom)] + [largest]
    keywords = [min(5, len(NUTRITION.findall(text))) / 5, min(5, len(MEASURE.findall(text))) / 5,
                min(5, len(PROMOTION.findall(text))) / 5, text.count("%") / 5, min(5, len(SEAL.findall(text))) / 5]
    whole_language = [float(whole == k) for k in LANGUAGES + ["other"]]
    values = amount + languages[:7] + languages[7:] + position + keywords + whole_language
    return np.array(values[:FEATURE_COUNT] + [0.0] * (FEATURE_COUNT - len(values)), dtype=np.float32), boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, default=Path("results/ocr"))
    parser.add_argument("--parts", nargs="+", default=["holdout", "train", "dev"])
    parser.add_argument("--image-size", type=int, default=1024, help="images scaled to about this squared for OCR")
    parser.add_argument("--limit", type=int, default=0)
    arguments = parser.parse_args()

    import easyocr

    reader = easyocr.Reader(["de", "en", "fr", "it", "nl"], gpu=torch.cuda.is_available(), verbose=False)
    detector = language_detector()
    parts = load_parts(arguments.data_dir)
    for part in arguments.parts:
        rows = parts[part] if arguments.limit == 0 else parts[part].head(arguments.limit)
        jobs = [(image_id, row, mirrored) for image_id, row, mirrored, with_context
                in extraction_jobs(part, rows, mirror=True, no_context_share=0.0) if with_context]
        part_dir = arguments.features_dir / part
        part_dir.mkdir(parents=True, exist_ok=True)
        done, features, started = {}, [], time.time()
        with (part_dir / "ocr.jsonl").open("w") as raw:
            for k, (image_id, row, mirrored) in enumerate(jobs):
                if image_id not in done:  # a mirrored copy has no text: it takes the original's features
                    with Image.open(row["image_file"]) as image:
                        rgb = common.opaque_rgb_on_white(image)
                    scaled = common.resize_to_area(rgb, arguments.image_size, multiple=1, max_side_factor=2)
                    result = reader.readtext(np.asarray(scaled))
                    done[image_id], boxes = features_of(result, scaled.size, detector)
                    raw.write(json.dumps({"image_id": str(image_id), "size": list(scaled.size), "boxes": boxes}) + "\n")
                features.append(done[image_id])
                if (k + 1) % 500 == 0:
                    print(f"{part}: {k + 1} / {len(jobs)} | {(k + 1) / (time.time() - started):.1f} images/s", flush=True)
        np.savez(part_dir / "shard_0000.npz", image_ids=np.array([j[0] for j in jobs]),
                 mirrored=np.array([j[2] for j in jobs]), context=np.ones(len(jobs), dtype=bool),
                 features=np.stack(features)[:, None, None], layers=np.array([0]))
        print(f"{part}: {len(jobs)} images, {len(jobs) / (time.time() - started):.1f} images/s", flush=True)


if __name__ == "__main__":
    main()
