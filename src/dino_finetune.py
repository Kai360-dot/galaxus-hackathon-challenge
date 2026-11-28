"""Finetune DINOv2-large for the visual dimensions (perspective, subject, text_classification), on one 12 GB GPU.

    python src/dino_finetune.py train --data-dir ~/galaxus/data --output-dir runs/dino_perspective
    python src/dino_finetune.py predict --data-dir ~/galaxus/data --output-dir runs/dino_perspective --parts holdout dev
    python src/dino_finetune.py predict ... --csv final/final.csv --name final

Train: images with a human, relevant perspective label from the train part (holdout untouched); 10% of the train
products are held back to pick the best epoch. The last blocks of DINOv2 and a new head learn; the rest stays frozen.
Augmentations keep the perspective: mirroring with left and right swapped, a mild crop, colour, a small rotation.
Each image is letterboxed onto white at SIZE x SIZE (aspect ratio kept). Prediction averages the image and its mirror
(left and right swapped back for perspective).

--dimensions trains one head per listed dimension on a shared backbone (each image learns from the labels it has).
--half 0|1 trains on one half of the train products and predicts the other half: the two halves together give every
train image a prediction from a model that never saw it, as input features for the Apertus heads (encoder outputs).
Predictions are saved as image_id -> dimension -> probabilities in common.CLASSES order.
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

import common
from heads import MIRRORED_PERSPECTIVES, load_parts

CLASSES = common.CLASSES["perspective"]
SWAP = [CLASSES.index(MIRRORED_PERSPECTIVES.get(c, c)) for c in CLASSES]
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def letterbox(rgb, size):
    rgb = rgb.copy()
    rgb.thumbnail((size, size), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(rgb, ((size - rgb.width) // 2, (size - rgb.height) // 2))
    return canvas


def to_tensor(rgb):
    return torch.from_numpy(((np.asarray(rgb, dtype=np.float32) / 255.0 - MEAN) / STD).astype(np.float32)).permute(2, 0, 1)


class Images(Dataset):
    def __init__(self, items, size, train):
        self.items, self.size, self.train = items, size, train  # items: (image_file, labels: one index or -1 per dimension)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_file, labels = self.items[index]
        labels = list(labels)
        with Image.open(image_file) as image:
            rgb = common.opaque_rgb_on_white(ImageOps.exif_transpose(image))
        if self.train:
            if random.random() < 0.5:  # mirror: left and right swap, front, back, top and bottom stay
                rgb = ImageOps.mirror(rgb)
                if PERSPECTIVE is not None and labels[PERSPECTIVE] >= 0:
                    labels[PERSPECTIVE] = SWAP[labels[PERSPECTIVE]]
            w, h = rgb.size
            share = random.uniform(0.75, 1.0)
            cw, ch = max(1, int(w * share ** 0.5)), max(1, int(h * share ** 0.5))
            x, y = random.randint(0, w - cw), random.randint(0, h - ch)
            rgb = rgb.crop((x, y, x + cw, y + ch))
            rgb = rgb.rotate(random.uniform(-8, 8), resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
            array = np.asarray(rgb, dtype=np.float32)
            array = array * random.uniform(0.8, 1.2) + random.uniform(-20, 20)  # contrast and brightness
            rgb = Image.fromarray(array.clip(0, 255).astype(np.uint8))
        return to_tensor(letterbox(rgb, self.size)), torch.tensor(labels)


DIMENSIONS = ["perspective"]  # set from --dimensions
PERSPECTIVE = 0


class Model(torch.nn.Module):
    def __init__(self, trainable_blocks):
        super().__init__()
        from transformers import AutoModel

        self.dino = AutoModel.from_pretrained("facebook/dinov2-large")
        for parameter in self.dino.parameters():
            parameter.requires_grad = False
        for block in self.dino.encoder.layer[-trainable_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in self.dino.layernorm.parameters():
            parameter.requires_grad = True
        # non-reentrant: the frozen blocks before pass no gradient, the trainable ones must still get theirs
        self.dino.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.heads = torch.nn.ModuleList(torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(2048, len(common.CLASSES[d])))
                                         for d in DIMENSIONS)

    def forward(self, pixels):
        hidden = self.dino(pixel_values=pixels).last_hidden_state
        pooled = torch.cat([hidden[:, 0], hidden[:, 1:].mean(1)], dim=1)
        return [head(pooled) for head in self.heads]


@torch.inference_mode()
def probabilities(model, items, size, batch, workers):
    """Per dimension: mean of the image and its mirror (perspective swapped back), softmax, in CLASSES order."""
    model.eval()
    loader = DataLoader(Images(items, size, train=False), batch_size=batch, num_workers=workers)
    out = [[] for _ in DIMENSIONS]
    for pixels, _ in loader:
        pixels = pixels.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            plain = model(pixels)
            mirrored = model(torch.flip(pixels, dims=[3]))
        for k, d in enumerate(DIMENSIONS):
            m = torch.softmax(mirrored[k].float(), 1)
            if d == "perspective":
                m = m[:, SWAP]
            out[k].append(((torch.softmax(plain[k].float(), 1) + m) / 2).cpu().numpy())
    return [np.concatenate(o) for o in out]


def macro_f1(labels, predicted):
    f1 = []
    for c in set(labels) | set(predicted):
        tp = np.sum((predicted == c) & (labels == c))
        fp = np.sum((predicted == c) & (labels != c))
        fn = np.sum((predicted != c) & (labels == c))
        f1.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f1))


def label_index(row, d):
    scored = row[f"{d}_source"] == "human" and row[f"{d}_relevant"].lower() == "true"
    return common.CLASSES[d].index(row[d]) if scored else -1


def items_of(rows):
    """(image_file, labels) for images with at least one usable label."""
    items = [(r["image_file"], [label_index(r, d) for d in DIMENSIONS]) for r in rows.to_dict("records")]
    return [(f, l) for f, l in items if max(l) >= 0]


def halves(rows):
    """Train products split in two halves, the same for every run."""
    products = sorted(rows["product_id"].unique())
    random.Random(common.SEED + 1).shuffle(products)
    return set(products[::2]), set(products[1::2])


def scores(items, p):
    return {d: macro_f1(np.array([l[k] for _, l in items if l[k] >= 0]), p[k][[l[k] >= 0 for _, l in items]].argmax(1))
            for k, d in enumerate(DIMENSIONS)}


def train(arguments):
    torch.manual_seed(common.SEED)
    random.seed(common.SEED)
    rows = load_parts(arguments.data_dir)["train"]
    if arguments.half is not None:
        rows = rows[rows["product_id"].isin(halves(rows)[arguments.half])]
    products = sorted(rows["product_id"].unique())
    random.Random(common.SEED).shuffle(products)
    held = set(products[: len(products) // 10])
    fit = items_of(rows[~rows["product_id"].isin(held)])
    val_rows = rows[rows["product_id"].isin(held)]
    val = items_of(val_rows)
    print(f"dimensions {DIMENSIONS} | train {len(fit)} images, validation {len(val)} images", flush=True)

    model = Model(arguments.trainable_blocks).cuda()
    losses_by_dim = []
    for k, d in enumerate(DIMENSIONS):
        counts = torch.bincount(torch.tensor([l[k] for _, l in fit if l[k] >= 0]), minlength=len(common.CLASSES[d])).float()
        weights = (counts.clamp(min=1) ** -0.5) * (counts > 0)
        weights = (weights / (weights * counts).sum() * counts.sum()).cuda()
        losses_by_dim.append(torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1, ignore_index=-1))
    backbone = [p for p in model.dino.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": arguments.lr},
                                   {"params": model.heads.parameters(), "lr": arguments.head_lr}], weight_decay=0.05)
    loader = DataLoader(Images(fit, arguments.size, train=True), batch_size=arguments.batch, shuffle=True,
                        num_workers=arguments.workers, drop_last=True, persistent_workers=True)
    total = arguments.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[arguments.lr, arguments.head_lr], total_steps=total,
                                                    pct_start=0.1, anneal_strategy="cos")
    scaler = torch.amp.GradScaler()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(arguments.epochs):
        model.train()
        started, losses = time.time(), []
        for step, (pixels, labels) in enumerate(loader):
            labels = labels.cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model(pixels.cuda(non_blocking=True))
            loss = sum(f(o.float(), labels[:, k]) for k, (f, o) in enumerate(zip(losses_by_dim, outputs))
                       if (labels[:, k] >= 0).any())
            if not torch.is_tensor(loss):  # no label of any dimension in this batch
                continue
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
            if step % 50 == 0:
                print(f"epoch {epoch + 1} step {step}/{len(loader)} loss {np.mean(losses[-50:]):.3f} "
                      f"| {(step + 1) * arguments.batch / (time.time() - started):.1f} images/s", flush=True)
        f1 = scores(val, probabilities(model, val, arguments.size, arguments.batch, arguments.workers))
        mean = float(np.mean(list(f1.values())))
        print(f"== epoch {epoch + 1}: loss {np.mean(losses):.3f} | validation macro F1 "
              + " ".join(f"{d} {v:.4f}" for d, v in f1.items()) + f" | mean {mean:.4f}", flush=True)
        if mean > best:
            best = mean
            torch.save(model.state_dict(), arguments.output_dir / "best.pt")
    print(f"best validation mean macro F1 {best:.4f}", flush=True)


def predict(arguments):
    model = Model(arguments.trainable_blocks).cuda()
    model.load_state_dict(torch.load(arguments.output_dir / "best.pt", map_location="cuda"))
    parts = load_parts(arguments.data_dir)
    targets = {name: parts[name] for name in arguments.parts}
    if arguments.half is not None:  # the train half this model did not see
        train = parts["train"]
        targets["other_half"] = train[train["product_id"].isin(halves(train)[1 - arguments.half])]
    if arguments.csv:
        targets[arguments.name] = common.read_part(arguments.csv)
    for name, rows in targets.items():
        started = time.time()
        p = probabilities(model, [(f, [-1] * len(DIMENSIONS)) for f in rows["image_file"]], arguments.size,
                          arguments.batch, arguments.workers)
        result = {i: {d: p[k][n] for k, d in enumerate(DIMENSIONS)} for n, i in enumerate(rows["image_id"])}
        np.save(arguments.output_dir / f"{name}_probabilities.npy", result, allow_pickle=True)
        if name in ("holdout", "other_half"):
            items = [(f, [label_index(r, d) for d in DIMENSIONS]) for f, r in zip(rows["image_file"], rows.to_dict("records"))]
            keep = [n for n, (_, l) in enumerate(items) if max(l) >= 0]
            f1 = scores([items[n] for n in keep], [q[keep] for q in p])
            print(f"{name} macro F1 (DINOv2 finetuned alone): " + " ".join(f"{d} {v:.4f}" for d, v in f1.items()), flush=True)
        print(f"{name}: {len(rows)} images in {time.time() - started:.0f} s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dino_perspective"))
    parser.add_argument("--size", type=int, default=336)
    parser.add_argument("--trainable-blocks", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--parts", nargs="*", default=["holdout", "dev"])
    parser.add_argument("--csv", type=Path, help="predict: another csv of images (for example final.csv)")
    parser.add_argument("--name", default="final")
    parser.add_argument("--dimensions", nargs="+", default=["perspective"])
    parser.add_argument("--half", type=int, choices=[0, 1], help="train on one half of the train products")
    arguments = parser.parse_args()
    global DIMENSIONS, PERSPECTIVE
    DIMENSIONS = arguments.dimensions
    PERSPECTIVE = DIMENSIONS.index("perspective") if "perspective" in DIMENSIONS else None
    train(arguments) if arguments.command == "train" else predict(arguments)


if __name__ == "__main__":
    main()
