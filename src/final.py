"""The final submission: heads trained on every labeled image (train + holdout), predictions for dev and final.

    python src/final.py --data-dir ~/galaxus/data \
        --heads results/heads_512/features=24:both:1024=results/dino_448_mirrored,results/pixels \
        --mirror-dirs results/dino_448_mirrored,results/pixels_mirrored \
        [--tuned runs/dino_multi_h0 runs/dino_multi_h1] --output-dir outputs/final

Same model as src/oof.py measured: six MLP heads on Apertus layer 24 (last token + image-token mean), DINOv2-large at
448 px and the pixel measures, each the mean of 5 seeds; the perspective is averaged with a mirror head on DINOv2 and
pixel features of the image and its mirror. --tuned adds the outputs of the finetuned DINOv2 as inputs. Writes
submission.csv with every dev and final row, and the class probabilities. The final features must exist under the
same folders as part "final" (data/final/final.csv decrypted, then the extraction scripts run with --parts final).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import common
from ensemble import argmax_answers, extra_features, to_frame, write_submission
from heads import load_features, load_parts, predict, targets
from oof import SWAP, features, mix_perspective, train_heads


def stack(parts_data):
    ids, mirrored, context, x = zip(*parts_data)
    return np.concatenate(ids), np.concatenate(mirrored), np.concatenate(context), np.vstack(x)


def mirror_features(dirs, part):
    ids, mirrored = load_features(Path(dirs[0]), part)[:2]
    return ids, mirrored, np.hstack([extra_features(d, part, ids, mirrored) for d in dirs])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--heads", required=True)
    parser.add_argument("--mirror-dirs", required=True)
    parser.add_argument("--mirror-weight", type=float, default=0.5)
    parser.add_argument("--tuned", nargs="*")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--parts", nargs="+", default=["dev", "final"], help="parts to predict")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    options = argparse.Namespace(weight_power=0.5, sampling=0.0, rare_noise=0.0, mixup=0.0)

    parts = load_parts(arguments.data_dir)
    labeled = pd.concat([parts["train"], parts["holdout"]], ignore_index=True)
    product_of = {i: p for part in parts.values() for i, p in zip(part["image_id"], part["product_id"])}
    hidden = int(arguments.heads.split("=")[1].split(":")[2])

    # Main heads on every labeled feature row (train with its mirrored and context-free copies, holdout with both).
    ids, mirrored, _, x = stack([features(arguments.heads, part, product_of, False, arguments.tuned) for part in ("train", "holdout")])
    y = targets(labeled, ids, mirrored)
    targets_x = {}
    for part in arguments.parts:
        pids, pmir, pctx, px = features(arguments.heads, part, product_of, False, arguments.tuned)
        keep = pctx & ~pmir
        targets_x[part] = (pids[keep], px[keep])
    print(f"training rows {len(x)} x {x.shape[1]} | " + " | ".join(f"{p} {len(v[0])}" for p, v in targets_x.items()), flush=True)
    sums = {p: None for p in targets_x}
    for seed in range(arguments.seeds):
        heads, mean, std = train_heads(x, y, hidden, arguments.epochs, arguments.device, common.SEED + seed, options)
        for p, (_, px) in targets_x.items():
            q = predict(heads, mean, std, px, arguments.device)
            sums[p] = q if sums[p] is None else {d: sums[p][d] + q[d] for d in q}
        del heads
        torch.cuda.empty_cache()
        print(f"main heads seed {seed + 1}/{arguments.seeds}", flush=True)
    probabilities = {p: to_frame(targets_x[p][0], {d: v / arguments.seeds for d, v in sums[p].items()}) for p in targets_x}

    # Mirror head for perspective on DINOv2 + pixels, image and mirror, swapped back.
    dirs = arguments.mirror_dirs.split(",")
    mids, mmir, mx = zip(*[mirror_features(dirs, part) for part in ("train", "holdout")])
    mids, mmir, mx = np.concatenate(mids), np.concatenate(mmir), np.vstack(mx)
    my = targets(labeled, mids, mmir)
    mirror_sum = {p: {} for p in targets_x}
    loaded = {p: mirror_features(dirs, p) for p in targets_x}
    for seed in range(arguments.seeds):
        heads, mean, std = train_heads(mx, my, hidden, arguments.epochs, arguments.device, common.SEED + 100 + seed, options, ["perspective"])
        for p, (pids, pmir, px) in loaded.items():
            q = predict(heads, mean, std, px, arguments.device)["perspective"]
            for row, (i, m) in enumerate(zip(pids, pmir)):
                mirror_sum[p][i] = mirror_sum[p].get(i, 0) + (q[row][SWAP] if m else q[row])
        print(f"mirror head seed {seed + 1}/{arguments.seeds}", flush=True)
    answers = {}
    for p in targets_x:
        mirror = {i: v / v.sum() for i, v in mirror_sum[p].items()}
        probabilities[p] = mix_perspective(probabilities[p], mirror, arguments.mirror_weight)
        np.save(arguments.output_dir / f"{p}_probabilities.npy", probabilities[p], allow_pickle=True)
        answers.update(argmax_answers(probabilities[p]))
    sample = pd.read_csv(arguments.data_dir / "sample_submission.csv", dtype=str)
    missing = set(sample["image_id"]) - set(answers)
    print(f"predicted {len(answers)} of {len(sample)} submission rows, {len(missing)} keep the sample constants", flush=True)
    write_submission(arguments.data_dir, answers, arguments.output_dir / "submission.csv")


if __name__ == "__main__":
    main()
