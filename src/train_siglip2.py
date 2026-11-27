import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from metric import DIMENSIONS, VALID_CLASSES

SIGLIP_ID = "google/siglip2-so400m-patch14-384"
LABEL_MAPS = {d: sorted(VALID_CLASSES[d]) for d in DIMENSIONS}
IDX_MAPS = {d: {c: i for i, c in enumerate(LABEL_MAPS[d])} for d in DIMENSIONS}


class GalaxusVisionDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str, processor, is_train: bool = True):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.is_train = is_train
        self.items = []
        for _, r in self.df.iterrows():
            m = list(self.img_dir.glob(f"{r['image_id']}.*"))
            if m:
                self.items.append((m[0], r))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, r = self.items[idx]
        img = Image.open(path).convert("RGB")
        px = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        if not self.is_train:
            return px, str(r["image_id"])
        targets = {}
        for d in DIMENSIONS:
            val = str(r[d]).strip() if pd.notna(r[d]) else ""
            targets[d] = IDX_MAPS[d].get(val, -100)
        return px, targets


class SigLIPMultiHead(nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_id).vision_model
        hidden = self.backbone.config.hidden_size
        self.heads = nn.ModuleDict({
            d: nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(0.2), nn.Linear(hidden, len(LABEL_MAPS[d])))
            for d in DIMENSIONS
        })

    def forward(self, pixel_values):
        feat = self.backbone(pixel_values=pixel_values).pooler_output
        return {d: head(feat) for d, head in self.heads.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/train_clean.csv")
    parser.add_argument("--train-img", default="data/train_images")
    parser.add_argument("--val-csv", default="data/synthetic_test.csv")
    parser.add_argument("--val-img", default="data/synthetic_images")
    parser.add_argument("--out-pred", default="data/siglip2_synthetic_preds.csv")
    parser.add_argument("--out-probs", default="data/siglip2_synthetic_probs.npz")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(SIGLIP_ID)
    train_ds = GalaxusVisionDataset(args.train_csv, args.train_img, processor, is_train=True)
    val_ds = GalaxusVisionDataset(args.val_csv, args.val_img, processor, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = SigLIPMultiHead(SIGLIP_ID).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 2e-5},
        {"params": model.heads.parameters(), "lr": 5e-4},
    ], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.amp.GradScaler("cuda")

    for ep in range(args.epochs):
        model.train()
        total_loss = 0.0
        for px, targets in train_loader:
            px = px.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(px)
                loss = sum(criterion(logits[d], targets[d].to(device)) for d in DIMENSIONS)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        print(f"Epoch {ep+1}/{args.epochs} | Avg Loss: {total_loss / max(1, len(train_loader)):.4f}")

    model.eval()
    preds = []
    prob_buffers = {d: [] for d in DIMENSIONS}
    ordered_ids = []

    with torch.no_grad():
        for px, img_ids in val_loader:
            px = px.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(px)
            ordered_ids.extend(img_ids)
            for d in DIMENSIONS:
                probs = torch.softmax(logits[d].float(), dim=-1).cpu().numpy()
                prob_buffers[d].append(probs)

    stacked_probs = {d: np.concatenate(prob_buffers[d], axis=0) for d in DIMENSIONS}
    for i, img_id in enumerate(ordered_ids):
        row = {"image_id": img_id}
        for d in DIMENSIONS:
            best_idx = int(stacked_probs[d][i].argmax())
            row[d] = LABEL_MAPS[d][best_idx]
        preds.append(row)

    val_df = pd.read_csv(args.val_csv)
    out_df = val_df[["image_id"]].merge(pd.DataFrame(preds), on="image_id", how="left")
    out_df.to_csv(args.out_pred, index=False)
    np.savez_compressed(args.out_probs, image_ids=np.array(ordered_ids), **stacked_probs)
    print(f"Saved SigLIP-2 predictions to {args.out_pred} and probabilities to {args.out_probs}")


if __name__ == "__main__":
    main()