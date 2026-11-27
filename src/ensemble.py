import argparse
import numpy as np
import pandas as pd
from metric import DIMENSIONS, VALID_CLASSES

LABEL_MAPS = {d: sorted(VALID_CLASSES[d]) for d in DIMENSIONS}
IDX_MAPS = {d: {c: i for i, c in enumerate(LABEL_MAPS[d])} for d in DIMENSIONS}

# Dimension-aware weights for Apertus-v1.5-8B (stronger on text/language) vs SigLIP-2 (stronger on framing)
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
    parser.add_argument("--siglip-probs", required=True, help="Path to siglip2_*_probs.npz")
    parser.add_argument("--apertus-csv", required=True, help="Path to Apertus CSV predictions")
    parser.add_argument("--out-csv", required=True, help="Path to blended raw CSV")
    args, _ = parser.parse_known_args()

    npz = np.load(args.siglip_probs, allow_pickle=True)
    siglip_ids = [str(x) for x in npz["image_ids"]]
    siglip_idx = {img_id: i for i, img_id in enumerate(siglip_ids)}

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
            w_ap = DEFAULT_APERTUS_WEIGHTS[d]
            ap_vec = np.zeros_like(probs)
            if ap_label in IDX_MAPS[d]:
                ap_vec[IDX_MAPS[d][ap_label]] = 1.0
            else:
                w_ap = 0.0

            combined = (1.0 - w_ap) * probs + w_ap * ap_vec
            out_row[d] = LABEL_MAPS[d][int(combined.argmax())]

        blended_rows.append(out_row)

    pd.DataFrame(blended_rows).to_csv(args.out_csv, index=False)
    print(f"Saved blended ensemble predictions to {args.out_csv}")


if __name__ == "__main__":
    main()