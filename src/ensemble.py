import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from metric import DIMENSIONS, VALID_CLASSES

LABEL_MAPS = {d: sorted(VALID_CLASSES[d]) for d in DIMENSIONS}
IDX_MAPS = {d: {c: i for i, c in enumerate(LABEL_MAPS[d])} for d in DIMENSIONS}

DEFAULT_APERTUS_WEIGHTS = {
    "subject": 0.50,
    "background": 0.45,
    "perspective": 0.40,
    "individuals": 0.50,
    "text_classification": 0.65,
    "text_language": 0.70,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--siglip-probs", required=True)
    parser.add_argument("--apertus-csv", required=True)
    parser.add_argument("--multipliers-json", default=None)
    parser.add_argument("--out-csv", required=True)
    args, _ = parser.parse_known_args()

    npz = np.load(args.siglip_probs, allow_pickle=True)
    siglip_ids = [str(x) for x in npz["image_ids"]]
    siglip_idx = {img_id: i for i, img_id in enumerate(siglip_ids)}

    mults = None
    if args.multipliers_json and Path(args.multipliers_json).exists():
        with open(args.multipliers_json, "r", encoding="utf-8") as f:
            raw_mults = json.load(f)
        mults = {
            d: np.array([raw_mults[d].get(c, 1.0) for c in LABEL_MAPS[d]], dtype=np.float32)
            for d in DIMENSIONS
        }

    ap_df = pd.read_csv(args.apertus_csv)
    blended_rows = []

    for _, row in ap_df.iterrows():
        img_id = str(row["image_id"])
        out_row = {"image_id": img_id}
        s_i = siglip_idx.get(img_id)

        for d in DIMENSIONS:
            ap_label = str(row[d]).strip() if pd.notna(row[d]) else ""
            if s_i is None:
                out_row[d] = ap_label if ap_label in IDX_MAPS[d] else LABEL_MAPS[d][0]
                continue

            probs = npz[d][s_i].copy()
            if mults is not None:
                probs = probs * mults[d]
                probs = probs / max(probs.sum(), 1e-8)

            w_ap = DEFAULT_APERTUS_WEIGHTS[d]
            ap_vec = np.zeros_like(probs)
            if ap_label in IDX_MAPS[d]:
                ap_vec[IDX_MAPS[d][ap_label]] = 1.0
            else:
                w_ap = 0.0

            combined = (1.0 - w_ap) * probs + w_ap * ap_vec
            out_row[d] = LABEL_MAPS[d][int(combined.argmax())]

        # Enforce cross-dimension consistency
        if out_row["text_classification"] in {"none", "brand_logo_only"}:
            out_row["text_language"] = "none"
        elif out_row["text_language"] == "none":
            out_row["text_language"] = "english"

        blended_rows.append(out_row)

    pd.DataFrame(blended_rows).to_csv(args.out_csv, index=False)
    print(f"Saved ensemble predictions to {args.out_csv}")


if __name__ == "__main__":
    main()