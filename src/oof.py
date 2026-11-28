"""Out-of-fold predictions over the whole train part, to tune class biases on 11,000 images instead of 600.

    python src/oof.py --data-dir ~/galaxus/data \
        --heads results/heads_512/features=24:both:1024=results/dino_448_mirrored,results/pixels \
        --folds 5 --seeds 5 --output-dir outputs/oof

The train part is split into folds of whole products. For each fold, heads are trained on the other folds and predict
the fold's images (with context, not mirrored), so every train image gets a prediction from heads that never saw it.
Class biases (an offset per class on the log-probabilities) are tuned on these out-of-fold predictions and checked on
the holdout, which neither the heads nor the biases have seen. The dev submission uses heads trained on the whole
train part, with and without the biases.

--product-mean appends to each image the mean features of all images of its product (product context).

Training variants (all on the heads only, the saved features stay as they are):
  --weight-power P   loss weight of a class = count^-P (0.5 so far; 1 weighs by inverse prevalence)
  --sampling Q       draw training rows with probability ~ count^-Q of their class (0 = plain shuffling)
  --rare-noise S     feature-space augmentation: Gaussian noise of S standard deviations on rows of rare classes
  --mixup P          same-class mixup: a share P of the rows (rare classes twice as often) is replaced by a random blend
                     of itself and another row with the same label, so the label stays right by construction
  --mirror-perspective W  a second perspective head on DINOv2 + pixel features, asked on the image and on its mirror
                     (left and right swapped back), mixed into the perspective probabilities with weight W
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

import common
from ensemble import apply_biases, argmax_answers, extra_features, line, score, to_frame, tune_biases, write_submission
from heads import MIRRORED_PERSPECTIVES, load_features, load_parts, make_head, predict, select, targets


TUNED_DIMENSIONS = ["perspective", "subject", "text_classification"]


def tuned_probabilities(run_dirs, part):
    """Encoder outputs of the finetuned DINOv2 (dino_finetune.py --half 0 and 1): image_id -> 29 numbers.

    Train images get the prediction of the half model that never saw them; other parts the mean of both.
    """
    runs = [{name: np.load(Path(d) / f"{name}_probabilities.npy", allow_pickle=True).item()
             for name in ("other_half", "holdout", "dev", "final") if (Path(d) / f"{name}_probabilities.npy").exists()} for d in run_dirs]
    flat = lambda p: np.concatenate([p[d] for d in TUNED_DIMENSIONS]).astype(np.float32)
    if part == "train":
        return {i: flat(p) for r in runs for i, p in r["other_half"].items()}
    return {i: np.mean([flat(r[part][i]) for r in runs], axis=0) for i in runs[0][part]}


def features(spec, part, product_of, product_mean, tuned_dirs=None):
    """Feature rows of one part: ids, mirrored, context flags and the matrix (Apertus + helper features)."""
    features_dir, configuration, *extra = spec.split("=")
    layers_text, pooling, _ = configuration.split(":")
    ids, mirrored, context, raw, layers = load_features(Path(features_dir), part)
    x = select(raw, [layers.index(int(layer)) for layer in layers_text.split(",")], pooling)
    for extra_dir in (extra[0].split(",") if extra else []):
        x = np.hstack([x, extra_features(extra_dir, part, ids, mirrored)])
    if tuned_dirs:  # finetuned DINOv2 outputs as extra inputs; for a mirrored row its perspective is swapped
        probs = tuned_probabilities(tuned_dirs, part)
        swap = np.array(SWAP)
        rows = []
        for i, m in zip(ids, mirrored):
            v = probs[i].copy()
            if m:
                v[:10] = v[:10][swap]
            rows.append(v)
        x = np.hstack([x, np.log(np.stack(rows) + 1e-4)])
    if product_mean:  # mean over the product's images as they are (with context, not mirrored)
        products = np.array([product_of[i] for i in ids])
        plain = context & ~mirrored
        means = {p: x[plain & (products == p)].mean(0) for p in set(products[plain])}
        x = np.hstack([x, np.stack([means.get(p, x[k]) for k, p in enumerate(products)])])
    return ids, mirrored, context, x


def train_heads(x, y, hidden, epochs, device, seed, options, dimensions=common.DIMENSIONS):
    """heads.train_heads with the variants: loss weight power, class-balanced sampling, noise on rare classes."""
    torch.manual_seed(seed)
    x = torch.tensor(x, device=device)
    mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True) + 1e-4
    x = (x - mean) / std
    heads = {}
    for dimension in dimensions:
        labels = y[dimension].to(device)
        keep = labels >= 0
        classes = len(common.CLASSES[dimension])
        counts = torch.bincount(labels[keep], minlength=classes).float()
        weights = (counts.clamp(min=1) ** -options.weight_power) * (counts > 0)
        weights = weights / (weights * counts).sum() * counts.sum()
        head = make_head(x.shape[1], hidden, classes).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.05)
        loss_function = torch.nn.CrossEntropyLoss(weight=weights)
        xs, ys = x[keep], labels[keep]
        draw = counts.clamp(min=1)[ys] ** -options.sampling  # per row, rare classes drawn more often
        rare = (counts[ys] < counts[counts > 0].median()).float()[:, None]  # rows of the rarer half of the classes
        by_class = torch.argsort(ys)  # rows sorted by class, to draw a partner with the same label
        starts = torch.cumsum(counts.long(), 0) - counts.long()
        for _ in range(epochs):
            order = (torch.multinomial(draw / draw.sum(), len(xs), replacement=True) if options.sampling
                     else torch.randperm(len(xs), device=device))
            for start in range(0, len(xs), 512):
                batch = order[start : start + 512]
                inputs = xs[batch]
                if options.mixup:
                    c = ys[batch]
                    partner = xs[by_class[starts[c] + (torch.rand(len(c), device=device) * counts[c]).long()]]
                    share = (options.mixup * (1 + rare[batch][:, 0])).clamp(max=1)
                    blend = (torch.rand(len(c), device=device) < share).float()[:, None] * torch.rand(len(c), 1, device=device)
                    inputs = (1 - blend) * inputs + blend * partner
                if options.rare_noise:
                    inputs = inputs + options.rare_noise * torch.randn_like(inputs) * rare[batch]
                optimizer.zero_grad()
                loss_function(head(inputs), ys[batch]).backward()
                optimizer.step()
        heads[dimension] = head.eval()
    return heads, mean, std


SWAP = [common.CLASSES["perspective"].index(MIRRORED_PERSPECTIVES.get(c, c)) for c in common.CLASSES["perspective"]]


def mirror_perspective(spec, part_rows, fit_ids, target_parts, hidden, epochs, seeds, device, options):
    """Perspective probabilities of a DINOv2 + pixel head, averaged over the image and its mirror (left/right swapped).

    target_parts: list of (part, image_ids). Returns one {image_id: probabilities} per target part.
    """
    extra_dirs = options.mirror_dirs.split(",") if options.mirror_dirs else spec.split("=")[2].split(",")
    def load(part):
        ids, mirrored = load_features(Path(extra_dirs[0]), part)[:2]
        x = np.hstack([extra_features(d, part, ids, mirrored) for d in extra_dirs])
        return ids, mirrored, x
    ids, mirrored, x = load("train")
    fit = np.isin(ids, list(fit_ids))
    y = targets(part_rows, ids[fit], mirrored[fit])
    loaded = {}
    sums = [dict() for _ in target_parts]
    for seed in range(seeds):
        heads, mean, std = train_heads(x[fit], y, hidden, epochs, device, common.SEED + 100 + seed, options, ["perspective"])
        for k, (part, wanted) in enumerate(target_parts):
            if part not in loaded:
                loaded[part] = load(part)
            pids, pmirrored, px = loaded[part]
            p = predict(heads, mean, std, px, device)["perspective"]
            for row, (i, m) in enumerate(zip(pids, pmirrored)):
                if i in wanted:
                    q = p[row][SWAP] if m else p[row]
                    sums[k][i] = sums[k].get(i, 0) + q
    return [{i: v / v.sum() for i, v in s.items()} for s in sums]


def mix_perspective(probabilities, mirror, weight):
    out = {}
    for i, p in probabilities.items():
        out[i] = dict(p)
        if i in mirror:
            out[i]["perspective"] = (1 - weight) * p["perspective"] + weight * mirror[i]
    return out


def mean_over_seeds(x_train, y, outputs, hidden, epochs, seeds, device, options):
    sums = [None] * len(outputs)
    for seed in range(seeds):
        heads, mean, std = train_heads(x_train, y, hidden, epochs, device, common.SEED + seed, options)
        for k, x in enumerate(outputs):
            p = predict(heads, mean, std, x, device)
            sums[k] = p if sums[k] is None else {d: sums[k][d] + p[d] for d in p}
        del heads
        torch.cuda.empty_cache()
    return [{d: v / seeds for d, v in s.items()} for s in sums]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--heads", required=True, help="FEATURES_DIR=LAYERS:POOLING:HIDDEN[=EXTRA_DIR[,EXTRA_DIR...]]")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--product-mean", action="store_true")
    parser.add_argument("--weight-power", type=float, default=0.5)
    parser.add_argument("--sampling", type=float, default=0.0)
    parser.add_argument("--rare-noise", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--context-only", action="store_true", help="train only on feature rows with product context")
    parser.add_argument("--mirror-perspective", type=float, default=0.0)
    parser.add_argument("--tuned", nargs="*", help="dino_finetune.py run folders (both halves) whose outputs become inputs")
    parser.add_argument("--mirror-dirs", help="helper feature folders with a mirrored copy of every image, comma separated")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oof"))
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    parts = load_parts(arguments.data_dir)
    metric = common.load_metric(arguments.data_dir)
    product_of = {i: p for part in parts.values() for i, p in zip(part["image_id"], part["product_id"])}
    hidden = int(arguments.heads.split("=")[1].split(":")[2])
    train_ids, train_mirrored, train_context, x_train = features(arguments.heads, "train", product_of, arguments.product_mean, arguments.tuned)
    hold_ids, hold_mirrored, hold_context, x_hold = features(arguments.heads, "holdout", product_of, arguments.product_mean, arguments.tuned)
    has_dev = any((Path(arguments.heads.split("=")[0]) / "dev").glob("shard_*.npz"))
    dev_ids, _, _, x_dev = (features(arguments.heads, "dev", product_of, arguments.product_mean, arguments.tuned) if has_dev
                            else (np.array([]), None, None, x_hold[:0]))
    if arguments.context_only:  # drop the copies without product context from training
        train_ids, train_mirrored, train_context, x_train = (a[train_context] for a in (train_ids, train_mirrored, train_context, x_train))
    keep = hold_context & ~hold_mirrored
    hold_ids, x_hold = hold_ids[keep], x_hold[keep]
    y = targets(parts["train"], train_ids, train_mirrored)
    print(f"train rows {len(x_train)} x {x_train.shape[1]} | holdout {len(x_hold)} | dev {len(dev_ids)}", flush=True)

    # Out-of-fold predictions over train products.
    products = sorted(set(product_of[i] for i in train_ids))
    random.Random(common.SEED).shuffle(products)
    fold_of = {p: k % arguments.folds for k, p in enumerate(products)}
    row_fold = np.array([fold_of[product_of[i]] for i in train_ids])
    plain = train_context & ~train_mirrored
    oof = {}
    for k in range(arguments.folds):
        fit = row_fold != k
        target = plain & (row_fold == k)
        y_fit = {d: v[torch.from_numpy(fit)] for d, v in y.items()}
        (p,) = mean_over_seeds(x_train[fit], y_fit, [x_train[target]], hidden, arguments.epochs, arguments.seeds, arguments.device, arguments)
        fold_p = to_frame(train_ids[target], p)
        if arguments.mirror_perspective:
            (mirror,) = mirror_perspective(arguments.heads, parts["train"], set(train_ids[fit]), [("train", set(train_ids[target]))],
                                           hidden, arguments.epochs, arguments.seeds, arguments.device, arguments)
            fold_p = mix_perspective(fold_p, mirror, arguments.mirror_perspective)
        oof.update(fold_p)
        print(f"fold {k + 1}/{arguments.folds}: {target.sum()} images predicted", flush=True)
    np.save(arguments.output_dir / "oof_probabilities.npy", oof, allow_pickle=True)
    oof_rows = parts["train"][parts["train"]["image_id"].isin(oof)].reset_index(drop=True)
    print(line("out-of-fold (train)", score(metric, oof_rows, oof)))

    # Heads on the whole train part for the holdout check and the dev submission.
    hold_p, dev_p = mean_over_seeds(x_train, y, [x_hold, x_dev], hidden, arguments.epochs, arguments.seeds, arguments.device, arguments)
    hold = to_frame(hold_ids, hold_p)
    dev = to_frame(dev_ids, dev_p)
    if arguments.mirror_perspective:
        mirror_hold, mirror_dev = mirror_perspective(
            arguments.heads, parts["train"], set(train_ids), [("holdout", set(hold_ids)), ("dev", set(dev_ids))],
            hidden, arguments.epochs, arguments.seeds, arguments.device, arguments)
        hold = mix_perspective(hold, mirror_hold, arguments.mirror_perspective)
        dev = mix_perspective(dev, mirror_dev, arguments.mirror_perspective)
    hold_rows = parts["holdout"][parts["holdout"]["image_id"].isin(hold)].reset_index(drop=True)
    biases = tune_biases(oof, oof_rows)
    print(line("out-of-fold + biases", score(metric, oof_rows, apply_biases(oof, biases))))
    print(line("holdout", score(metric, hold_rows, hold)))
    print(line("holdout + OOF biases", score(metric, hold_rows, apply_biases(hold, biases))))
    print("  biases: " + " | ".join(f"{d[:8]} " + ",".join(f"{c}={b:+.2f}" for c, b in zip(common.CLASSES[d], biases[d]) if b)
                                    for d in common.DIMENSIONS))
    np.save(arguments.output_dir / "biases.npy", biases, allow_pickle=True)
    np.save(arguments.output_dir / "holdout_probabilities.npy", hold, allow_pickle=True)
    np.save(arguments.output_dir / "dev_probabilities.npy", dev, allow_pickle=True)
    if has_dev:
        write_submission(arguments.data_dir, argmax_answers(dev), arguments.output_dir / "submission_plain.csv")
        write_submission(arguments.data_dir, argmax_answers(apply_biases(dev, biases)), arguments.output_dir / "submission_biases.csv")


if __name__ == "__main__":
    main()
