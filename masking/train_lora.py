# Apertus-v1.5-8B bf16 LoRA training
import argparse
import json
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForMultimodalLM, AutoProcessor, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

MODEL_ID = "swiss-ai/Apertus-v1.5-8B"
DIMENSIONS = ["subject", "background", "perspective", "individuals", "text_classification", "text_language"]

SYSTEM_PROMPT = (
    "Classify this e-commerce product image into JSON with exact keys: "
    "subject, background, perspective, individuals, text_classification, text_language. "
    "Respond ONLY with valid JSON."
)

class GalaxusMultimodalDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str, processor):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.records = []
        for _, row in self.df.iterrows():
            img_id = str(row["image_id"])
            matches = list(self.img_dir.glob(f"{img_id}.*"))
            if matches:
                self.records.append((matches[0], row))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_path, row = self.records[idx]
        image = Image.open(img_path).convert("RGB")
        target = {d: (str(row[d]).strip() if pd.notna(row[d]) else "none") for d in DIMENSIONS}
        target_json = json.dumps(target, ensure_ascii=False)

        prompt_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Classify this image."}]},
        ]
        full_msgs = prompt_msgs + [{"role": "assistant", "content": target_json}]

        prompt_text = self.processor.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        full_text = self.processor.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )

        prompt_inputs = self.processor(text=prompt_text, images=[image], return_tensors="pt")
        full_inputs = self.processor(text=full_text, images=[image], return_tensors="pt")

        input_ids = full_inputs["input_ids"][0]
        attention_mask = full_inputs["attention_mask"][0]
        labels = input_ids.clone()
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:prompt_len] = -100  # Mask prompt and all discrete <|image|> tokens

        item = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if "pixel_values" in full_inputs:
            item["pixel_values"] = full_inputs["pixel_values"][0]
        return item

def collate_fn(batch):
    max_len = max(x["input_ids"].shape[0] for x in batch)
    pad_id = 0
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, x in enumerate(batch):
        n = x["input_ids"].shape[0]
        input_ids[i, :n] = x["input_ids"]
        attention_mask[i, :n] = x["attention_mask"]
        labels[i, :n] = x["labels"]
    out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    if "pixel_values" in batch[0]:
        out["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/train_clean.csv")
    parser.add_argument("--img-dir", default="data/train_images")
    parser.add_argument("--out-dir", default="checkpoints/apertus-8b-lora")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    args, _ = parser.parse_known_args()

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # dtype="auto" keeps LLM in bf16 while preserving float32 on the EMU3.5 vision tokenizer
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = GalaxusMultimodalDataset(args.train_csv, args.img_dir, processor)
    train_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="none",
    )
    Trainer(model=model, args=train_args, train_dataset=dataset, data_collator=collate_fn).train()
    model.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)
    print(f"Saved LoRA adapter to {args.out_dir}")

if __name__ == "__main__":
    main()