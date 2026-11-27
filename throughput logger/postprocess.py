# Product-level consistency & sorting rules
import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from metric import DIMENSIONS, VALID_CLASSES

DEFAULTS = {
    "subject": "full_product",
    "background": "white",
    "perspective": "front_head_on",
    "individuals": "none",
    "text_classification": "none",
    "text_language": "none",
}

def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for dim in DIMENSIONS:
        out[dim] = out[dim].fillna(DEFAULTS[dim]).astype(str).str.strip()
        out.loc[~out[dim].isin(VALID_CLASSES[dim]), dim] = DEFAULTS[dim]

    # Cross-dimension consistency: no text means language must be 'none'
    no_text = out["text_classification"] == "none"
    out.loc[no_text, "text_language"] = "none"

    # If a specific language is detected, text_classification cannot remain 'none'
    has_lang = out["text_language"] != "none"
    out.loc[has_lang & no_text, "text_classification"] = "other"

    # Placeholders and standalone logos default to white background and front_head_on
    non_photo = out["subject"].isin({"placeholder", "logo"})
    out.loc[non_photo, "perspective"] = "front_head_on"

    return out[["image_id", *DIMENSIONS]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    args, _ = parser.parse_known_args()

    df = pd.read_csv(args.in_csv)
    cleaned = apply_rules(df)
    cleaned.to_csv(args.out_csv, index=False)
    print(f"Post-processed {len(cleaned)} rows -> {args.out_csv}")

if __name__ == "__main__":
    main()