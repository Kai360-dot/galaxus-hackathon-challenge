"""Shared pieces of the Galaxus pipeline: data, holdout split, image preparation, prompt, answer parsing, scoring.

Training and prediction both import this module, so they see exactly the same images and prompt.
"""

import io
import json
import random
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

DIMENSIONS = ("subject", "background", "perspective", "individuals", "text_classification", "text_language")

CLASSES = {
    "subject": [
        "full_product", "detail_closeup", "accessories", "packaging", "package_and_product", "multipack_trade_unit",
        "delivery_scope", "product_grouping", "multiview", "energy_label", "infographic", "logo", "placeholder",
    ],
    "background": ["white", "color", "scene_lifestyle"],
    "perspective": [
        "front_head_on", "front_angle_left", "front_angle_right", "back_head_on", "back_angle_left",
        "back_angle_right", "left_side", "right_side", "top_down", "bottom_up",
    ],
    "individuals": ["individual_present", "none"],
    "text_classification": [
        "certification_seal", "technical_specs", "promotional_text", "nutritional_info", "other", "none",
    ],
    "text_language": ["German", "English", "French", "Italian", "Dutch", "other", "none"],
}

DIMENSION_MEANINGS = {
    "subject": "what the image shows",
    "background": "the physical backdrop behind the product",
    "perspective": "from where the product is seen, judged by where its front points",
    "individuals": "a real person or animal in the photo",
    "text_classification": "text or elements added digitally on top of the photo, or a nutrition label",
    "text_language": "the dominant language of the visible text",
}

# The holdout of the reference notebook: whole products, so no image of a holdout product is in training.
HOLDOUT_IMAGES = 600
SEED = 3407


# ---------------------------------------------------------------- data


def read_part(csv_path):
    """A csv of the competition as strings, with the full path of each image."""
    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    frame["image_file"] = [str(csv_path.parent / image_path) for image_path in frame["image_path"]]
    return frame


def split_holdout(labeled_rows, holdout_images=HOLDOUT_IMAGES, seed=SEED):
    """Train and holdout rows, the holdout made of whole products (same split as the reference notebook)."""
    images_per_product = labeled_rows["product_id"].value_counts()
    shuffled_products = sorted(images_per_product.index)
    random.Random(seed).shuffle(shuffled_products)
    holdout_products, held_out = set(), 0
    for product_id in shuffled_products:
        if held_out >= holdout_images:
            break
        holdout_products.add(product_id)
        held_out += images_per_product[product_id]
    is_holdout = labeled_rows["product_id"].isin(holdout_products)
    return labeled_rows[~is_holdout].reset_index(drop=True), labeled_rows[is_holdout].reset_index(drop=True)


# ---------------------------------------------------------------- images


def opaque_rgb_on_white(image):
    """RGB, with any transparency composited onto white (a plain convert("RGB") turns it black)."""
    if image.mode not in ("RGBA", "LA", "PA") and "transparency" not in image.info:
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB")


# Apertus 1.5 makes one discrete vision token per 16 x 16 pixels (processor_config.json: spatial_factor 16),
# with at least 65,536 pixels per image. So 256 -> 256 tokens, 384 -> 576 tokens, 512 -> 1,024 tokens.
PATCH_SIZE = 16
MIN_IMAGE_SIZE = 256


def resize_to_area(image, image_size, multiple=PATCH_SIZE, max_side_factor=8):
    """The image scaled to about image_size**2 pixels, aspect ratio kept, each side a multiple of `multiple`."""
    scale = image_size / (image.width * image.height) ** 0.5
    scale *= min(1.0, max_side_factor * image_size / (max(image.width, image.height) * scale))
    width = max(multiple, round(image.width * scale / multiple) * multiple)
    height = max(multiple, round(image.height * scale / multiple) * multiple)
    return image.resize((width, height), Image.LANCZOS)


def prepare_image(image_file, image_size=512, jpeg_quality=92, mirrored=False):
    """The image as the model gets it: on white, about image_size**2 pixels, JPEG round trip."""
    with Image.open(image_file) as image:
        rgb = opaque_rgb_on_white(image)
    if mirrored:
        rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
    buffer = io.BytesIO()
    resize_to_area(rgb, image_size).save(buffer, format="JPEG", quality=jpeg_quality)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


# ---------------------------------------------------------------- prompt and answer


def system_prompt(guideline=None):
    """The task framing. A zero-shot model gets the labeling guideline on top, a finetuned one does not need it."""
    lines = [
        "You classify one product image of an online shop along six dimensions.",
        "Answer only with a JSON object that holds exactly one class per dimension, in this order:",
    ]
    lines += [f"- {d} ({DIMENSION_MEANINGS[d]}): {', '.join(CLASSES[d])}" for d in DIMENSIONS]
    if guideline:
        lines += ["", "Follow this labeling guideline:", "", guideline.strip()]
    return "\n".join(lines)


def user_prompt(row, with_context=True):
    """The product context of an image, or a bare instruction when the context is left out."""
    if not with_context:
        return "Classify this image."
    return (
        f"Product: {row['product_title']}\nBrand: {row['brand']}\nProduct type: {row['product_type']}\n"
        f"Category: {row['category']} / {row['subcategory']}"
    )


def parse_answer(answer):
    """The class per dimension of a model answer, None for a dimension it got wrong or left out."""
    try:
        parsed = json.loads(answer[answer.find("{") : answer.rfind("}") + 1])
    except ValueError:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    return {d: parsed.get(d) if parsed.get(d) in CLASSES[d] else None for d in DIMENSIONS}


# ---------------------------------------------------------------- scoring


def load_metric(data_dir):
    """metric.py from the competition files (it stays out of the repository)."""
    sys.path.insert(0, str(data_dir))
    import metric

    return metric


def score_rows(metric, rows, answers, fallback):
    """metric.evaluate on labeled rows: only human labels in relevant cells count, as on Kaggle."""
    solution = pd.DataFrame({"image_id": rows["image_id"]})
    for d in DIMENSIONS:
        scored = (rows[f"{d}_source"] == "human") & (rows[f"{d}_relevant"].str.lower() == "true")
        solution[d] = rows[d].where(scored)
    submission = pd.DataFrame(
        [
            {"image_id": image_id, **{d: (answers.get(image_id) or {}).get(d) or fallback[d] for d in DIMENSIONS}}
            for image_id in rows["image_id"]
        ]
    )
    return metric.evaluate(solution, submission, "image_id")


def print_metrics(metrics):
    """The per-dimension table and the weakest classes."""
    print(f"{'':20} {'F1 macro':>9} {'F1 weighted':>12} {'cells':>6}")
    for d in [d for d in DIMENSIONS if d in metrics] + ["overall"]:
        values = metrics[d]
        print(f"{d:20} {values['f1_macro']:9.3f} {values['f1_weighted']:12.3f} {values['support']:6}")
    weakest = sorted(
        (values["f1"], f"{d}/{name}", values["support"])
        for d in DIMENSIONS
        if d in metrics
        for name, values in metrics[d]["per_class"].items()
    )[:10]
    print("\nweakest classes:", ", ".join(f"{name} {f1:.2f} (n={n})" for f1, name, n in weakest))
