"""Paths, classes and settings shared by the pipeline."""

import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def find_upwards(name):
    """The first folder called name in the working directory or one of its parents."""
    for folder in [Path.cwd(), *Path.cwd().parents]:
        if (folder / name).is_dir():
            return folder / name
    raise FileNotFoundError(f"no {name} folder in {Path.cwd()} or above")


DATA_DIR = find_upwards("glx_data")
RUNS_DIR = DATA_DIR.parent / "runs"  # outside the repository: everything in it derives from competition images
STORE = RUNS_DIR / "question_tree"
STORE.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(DATA_DIR))
import metric  # noqa: E402, the leaderboard metric from the competition data

DIMENSIONS = metric.DIMENSIONS
CLASSES = {  # in the order of the labeling guideline
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
    "text_classification": ["nutritional_info", "technical_specs", "promotional_text", "certification_seal", "other", "none"],
    "text_language": ["German", "English", "French", "Italian", "Dutch", "other", "none"],
}
assert all(set(CLASSES[dimension]) == metric.VALID_CLASSES[dimension] for dimension in DIMENSIONS)

CONSTANT_ANSWER = {  # the most frequent class of every dimension, the floor any method has to beat
    "subject": "full_product",
    "background": "white",
    "perspective": "front_head_on",
    "individuals": "none",
    "text_classification": "none",
    "text_language": "none",
}

MODEL_8B = "apertus-v1-5-8b"
MODEL_70B = "apertus-v1-5-70b"
SEED = 0
IMAGE_SIZE = 512  # images are scaled to about IMAGE_SIZE x IMAGE_SIZE pixels in area
JPEG_QUALITY = 90
WORKERS = 24  # images handled in parallel; the questions of one image are sent one after the other
MAX_ATTEMPTS = 4
TIMEOUT_SECONDS = 120

# Speed option. The robust yes/no questions share one request, the fragile ones are asked alone.
# It saves more than half of the requests for a small loss. Set it to False if grouped replies cause trouble:
# every question is then asked alone and nothing else changes.
GROUP_ROBUST_QUESTIONS = True


def api_settings():
    """Base URL and key of the Apertus endpoint, read from the environment or a .env file."""
    load_dotenv(find_dotenv(usecwd=True))
    return os.environ["LLM_BASE_URL"].rstrip("/"), os.environ["LLM_API_KEY"]
