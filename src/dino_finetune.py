"""Finetune DINOv2-large for the perspective dimension, on one 12 GB GPU.

    python src/dino_finetune.py train --data-dir ~/galaxus/data --output-dir runs/dino_perspective
    python src/dino_finetune.py predict --data-dir ~/galaxus/data --output-dir runs/dino_perspective --parts holdout dev
    python src/dino_finetune.py predict ... --csv final/final.csv --name final

Train: images with a human, relevant perspective label from the train part (holdout untouched); 10% of the train
products are held back to pick the best epoch. The last blocks of DINOv2 and a new head learn; the rest stays frozen.
Augmentations keep the perspective: mirroring with left and right swapped, a mild crop, colour, a small rotation.
Each image is letterboxed onto white at SIZE x SIZE (aspect ratio kept). Prediction averages the image and its mirror
(left and right swapped back) and saves image_id -> 10 probabilities in common.CLASSES["perspective"] order.
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
        self.items, self.size, self.train = items, size, train  # items: (image_file, label index or -1)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_file, label = self.items[index]
        with Image.open(image_file) as image:
            rgb = common.opaque_rgb_on_white(ImageOps.exif_transpose(image))
        if self.train:
            if random.random() < 0.5:  # mirror: left and right swap, front, back, top and bottom stay
                rgb, label = ImageOps.mirror(rgb), SWAP[label]
            w, h = rgb.size
            share = random.uniform(0.75, 1.0)
            cw, ch = max(1, int(w * share ** 0.5)), max(1, int(h * share ** 0.5))
            x, y = random.randint(0, w - cw), random.randint(0, h - ch)
            rgb = rgb.crop((x, y, x + cw, y + ch))
            rgb = rgb.rotate(random.uniform(-8, 8), resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
            array = np.asarray(rgb, dtype=np.float32)
            array = array * random.uniform(0.8, 1.2) + random.uniform(-20, 20)  # contrast and brightness
            rgb = Image.fromarray(array.clip(0, 255).astype(np.uint8))
        return to_tensor(letterbox(rgb, self.size)), label


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
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(2048, len(CLASSES)))

    def forward(self, pixels):
        hidden = self.dino(pixel_values=pixels).last_hidden_state
        return self.head(torch.cat([hidden[:, 0], hidden[:, 1:].mean(1)], dim=1))


@torch.inference_mode()
def probabilities(model, items, size, batch, workers):
    """Mean of the image and its mirror (swapped back), softmax, in CLASSES order."""
    model.eval()
    loader = DataLoader(Images(items, size, train=False), batch_size=batch, num_workers=workers)
    out = []
    for pixels, _ in loader:
        pixels = pixels.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            plain = torch.softmax(model(pixels).float(), 1)
            mirrored = torch.softmax(model(torch.flip(pixels, dims=[3])).float(), 1)[:, SWAP]
        out.append(((plain + mirrored) / 2).cpu().numpy())
    return np.concatenate(out)


def macro_f1(labels, predicted):
    f1 = []
    for c in set(labels) | set(predicted):
        tp = np.sum((predicted == c) & (labels == c))
        fp = np.sum((predicted == c) & (labels != c))
        fn = np.sum((predicted != c) & (labels == c))
        f1.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f1))


def labeled(rows):
    keep = (rows["perspective_source"] == "human") & (rows["perspective_relevant"].str.lower() == "true")
    return rows[keep]


def train(arguments):
    torch.manual_seed(common.SEED)
    random.seed(common.SEED)
    rows = labeled(load_parts(arguments.data_dir)["train"])
    products = sorted(rows["product_id"].unique())
    random.Random(common.SEED).shuffle(products)
    held = set(products[: len(products) // 10])
    fit_rows, val_rows = rows[~rows["product_id"].isin(held)], rows[rows["product_id"].isin(held)]
    fit = [(f, CLASSES.index(p)) for f, p in zip(fit_rows["image_file"], fit_rows["perspective"])]
    val = [(f, CLASSES.index(p)) for f, p in zip(val_rows["image_file"], val_rows["perspective"])]
    print(f"train {len(fit)} images, validation {len(val)} images", flush=True)

    model = Model(arguments.trainable_blocks).cuda()
    counts = torch.bincount(torch.tensor([l for _, l in fit]), minlength=len(CLASSES)).float()
    weights = counts.clamp(min=1) ** -0.5
    weights = (weights / (weights * counts).sum() * counts.sum()).cuda()
    loss_function = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    backbone = [p for p in model.dino.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": arguments.lr},
                                   {"params": model.head.parameters(), "lr": arguments.head_lr}], weight_decay=0.05)
    loader = DataLoader(Images(fit, arguments.size, train=True), batch_size=arguments.batch, shuffle=True,
                        num_workers=arguments.workers, drop_last=True, persistent_workers=True)
    total = arguments.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[arguments.lr, arguments.head_lr], total_steps=total,
                                                    pct_start=0.1, anneal_strategy="cos")
    scaler = torch.amp.GradScaler()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    val_labels = np.array([l for _, l in val])
    for epoch in range(arguments.epochs):
        model.train()
        started, losses = time.time(), []
        for step, (pixels, labels) in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_function(model(pixels.cuda(non_blocking=True)).float(), labels.cuda())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
            if step % 50 == 0:
                print(f"epoch {epoch + 1} step {step}/{len(loader)} loss {np.mean(losses[-50:]):.3f} "
                      f"| {(step + 1) * arguments.batch / (time.time() - started):.1f} images/s", flush=True)
        p = probabilities(model, val, arguments.size, arguments.batch, arguments.workers)
        f1 = macro_f1(val_labels, p.argmax(1))
        print(f"== epoch {epoch + 1}: loss {np.mean(losses):.3f} | validation perspective macro F1 {f1:.4f}", flush=True)
        if f1 > best:
            best = f1
            torch.save({k: v for k, v in model.state_dict().items()}, arguments.output_dir / "best.pt")
            np.save(arguments.output_dir / "validation_probabilities.npy",
                    dict(zip(val_rows["image_id"], p)), allow_pickle=True)
    print(f"best validation macro F1 {best:.4f}", flush=True)


def predict(arguments):
    model = Model(arguments.trainable_blocks).cuda()
    model.load_state_dict(torch.load(arguments.output_dir / "best.pt", map_location="cuda"))
    parts = load_parts(arguments.data_dir)
    targets = {name: parts[name] for name in arguments.parts}
    if arguments.csv:
        targets[arguments.name] = common.read_part(arguments.csv)
    for name, rows in targets.items():
        started = time.time()
        p = probabilities(model, [(f, -1) for f in rows["image_file"]], arguments.size, arguments.batch, arguments.workers)
        np.save(arguments.output_dir / f"{name}_probabilities.npy", dict(zip(rows["image_id"], p)), allow_pickle=True)
        if name == "holdout":
            scored = labeled(rows)
            index = {i: k for k, i in enumerate(rows["image_id"])}
            f1 = macro_f1(np.array([CLASSES.index(c) for c in scored["perspective"]]),
                          np.array([p[index[i]].argmax() for i in scored["image_id"]]))
            print(f"holdout perspective macro F1 (DINOv2 finetuned alone) {f1:.4f}", flush=True)
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
    arguments = parser.parse_args()
    train(arguments) if arguments.command == "train" else predict(arguments)


if __name__ == "__main__":
    main()
