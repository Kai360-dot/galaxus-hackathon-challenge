import re
from typing import Tuple
import pandas as pd
from metric import DIMENSIONS, VALID_CLASSES

# Common VLM / annotation aliases mapped to canonical Galaxus classes
LABEL_ALIASES = {
    "subject": {
        "product": "full_product",
        "single_product": "full_product",
        "detail": "partial_product",
        "closeup": "partial_product",
        "close_up": "partial_product",
        "in_use": "product_in_use",
        "lifestyle": "product_in_use",
        "box": "packaging",
        "package": "packaging",
        "set": "bundle",
    },
    "background": {
        "pure_white": "white",
        "studio_white": "white",
        "transparent": "white",
        "solid": "solid_color",
        "grey": "solid_color",
        "gray": "solid_color",
        "black": "solid_color",
        "real_world": "lifestyle",
        "scene": "lifestyle",
        "studio": "studio_set",
    },
    "perspective": {
        "front": "front_head_on",
        "head_on": "front_head_on",
        "front_left": "front_angle_left",
        "three_quarter_left": "front_angle_left",
        "front_right": "front_angle_right",
        "three_quarter_right": "front_angle_right",
        "top": "top_down",
        "overhead": "top_down",
        "flat_lay": "top_down",
        "macro": "detail_macro",
    },
    "individuals": {
        "no": "none",
        "no_person": "none",
        "hand": "hands_only",
        "hands": "hands_only",
        "person": "full_person",
        "people": "multiple_people",
    },
    "text_classification": {
        "no_text": "none",
        "logo": "brand_logo_only",
        "brand_only": "brand_logo_only",
        "label": "product_name_or_label",
        "product_name": "product_name_or_label",
        "marketing": "marketing_callout",
        "specs": "technical_specs",
        "nutrition": "nutritional_info",
        "certification": "certification_seal",
    },
    "text_language": {
        "no_language": "none",
        "de": "german",
        "en": "english",
        "fr": "french",
        "it": "italian",
        "nl": "dutch",
        "multi": "multilingual",
    },
}

DEFAULT_FALLBACK = {
    "subject": "full_product",
    "background": "white",
    "perspective": "front_head_on",
    "individuals": "none",
    "text_classification": "none",
    "text_language": "none",
}


def normalize_token(val) -> str:
    if pd.isna(val):
        return ""
    s = str(val).strip().lower()
    s = re.sub(r"[\s\-/]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s


def canonicalize_label(dim: str, raw_val, allow_none_for_missing: bool = False) -> Tuple[str, bool]:
    """Returns (clean_label, is_valid_original)."""
    norm = normalize_token(raw_val)
    if not norm or norm in {"nan", "null", "na", "unknown"}:
        return ("" if allow_none_for_missing else DEFAULT_FALLBACK[dim]), False

    valid = VALID_CLASSES[dim]
    if norm in valid:
        return norm, True

    aliased = LABEL_ALIASES.get(dim, {}).get(norm)
    if aliased and aliased in valid:
        return aliased, True

    return ("" if allow_none_for_missing else DEFAULT_FALLBACK[dim]), False


def sanitize_dataframe(df: pd.DataFrame, is_ground_truth: bool = False) -> pd.DataFrame:
    """Cleans all 6 dimension columns and enforces cross-dimension consistency."""
    out = df.copy()
    anomaly_counts = {d: 0 for d in DIMENSIONS}

    for d in DIMENSIONS:
        if d not in out.columns:
            out[d] = "" if is_ground_truth else DEFAULT_FALLBACK[d]
            anomaly_counts[d] = len(out)
            continue

        cleaned = []
        for val in out[d]:
            lbl, ok = canonicalize_label(d, val, allow_none_for_missing=is_ground_truth)
            if not ok and pd.notna(val) and str(val).strip() != "":
                anomaly_counts[d] += 1
            cleaned.append(lbl)
        out[d] = cleaned

    # Cross-dimension consistency repair (for predictions and noisy labels)
    if not is_ground_truth and "text_classification" in out.columns and "text_language" in out.columns:
        no_text_mask = out["text_classification"].isin(["none", "brand_logo_only"])
        out.loc[no_text_mask, "text_language"] = "none"
        has_text_mask = (~no_text_mask) & (out["text_language"] == "none")
        out.loc[has_text_mask, "text_language"] = "english"

    for d, cnt in anomaly_counts.items():
        if cnt > 0:
            print(f"[label_sanitizer] Repaired {cnt} invalid/aliased entries in '{d}'")

    return out