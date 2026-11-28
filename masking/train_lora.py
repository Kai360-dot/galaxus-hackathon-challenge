import argparse
from collections import Counter
import json
import os
import sys
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from metric import DIMENSIONS
from prompts import SYSTEM_PROMPT
from label_sanitizer import sanitize_dataframe
from eval_dimensions import evaluate_predictions

MODEL_ID = "swiss-ai/Apertus-v1.5-8B"


class GalaxusMultimodalDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str, processor, rare_threshold: int = 150, oversample_factor: int = 2):
        raw_df = pd.read_csv(csv_path)
        self.df = sanitize_dataframe(raw_df, is_ground_truth=True)
        self.img_dir = Path(img_dir)
        self.processor = processor

        rare_labels = {d: set() for d in DIMENSIONS}
        for d in DIMENSIONS:
            counts = Counter(self.df[d][self.df[d] != ""])
            for cls, cnt in counts.items():
                if cnt < rare_threshold:
                    rare_labels[d].add(cls)

        self.records = []
        for _, row in self.df.iterrows():
            img_id = str(row["image_id"])
            matches = list(self.img_dir.glob(f"{img_id}.*"))
            if not matches:
                continue
            # Skip rows where every single dimension label was missing/corrupted
            if all(row[d] == "" for d in DIMENSIONS):
                continue
            has_rare = any(row[d] in rare_labels[d] for d in DIMENSIONS)
            repeats = oversample_factor if has_rare else 1
            for _ in range(repeats):
                self.records.append((matches[0], row))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_path, row = self.records[idx]
        image = Image.open(img_path).convert("RGB")
        target = {d: (row[d] if row[d] != "" else "none") for d in DIMENSIONS}
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
        labels[:prompt_len] = -100

        item = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if "pixel_values" in full_inputs:
            item["pixel_values"] = full_inputs["pixel_values"][0]
        return item


def collate_fn(batch):
    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
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


def run_batched_inference(model, processor, csv_path: str, img_dir: str, out_csv: str, batch_size: int = 24) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    img_root = Path(img_dir)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model.eval()
    rows = []
    for start in tqdm(range(0, len(df), batch_size), desc=f"Infer {Path(csv_path).name}"):
        chunk = df.iloc[start : start + batch_size]
        texts, images, valid_ids = [], [], []
        for _, r in chunk.iterrows():
            img_id = str(r["image_id"])
            matches = list(img_root.glob(f"{img_id}.*"))
            if not matches:
                rows.append({"image_id": img_id})
                continue
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Classify this image."}]},
            ]
            texts.append(processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            ))
            images.append(Image.open(matches[0]).convert("RGB"))
            valid_ids.append(img_id)

        if not texts:
            continue
        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            gen_ids = model.generate(
                **inputs, max_new_tokens=96, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id
            )
        for img_id, raw in zip(valid_ids, processor.batch_decode(gen_ids[:, prompt_len:], skip_special_tokens=True)):
            try:
                parsed = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
            except Exception:
                parsed = {}
            parsed["image_id"] = img_id
            rows.append(parsed)

    raw_preds = df[["image_id"]].merge(pd.DataFrame(rows), on="image_id", how="left")
    clean_preds = sanitize_dataframe(raw_preds, is_ground_truth=False)
    clean_preds.to_csv(out_csv, index=False)
    return clean_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/train_clean.csv")
    parser.add_argument("--img-dir", default="data/train_images")
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--val-img", default=None)
    parser.add_argument("--val-out-csv", default="outputs/apertus_val_preds.csv")
    parser.add_argument("--dev-csv", default=None)
    parser.add_argument("--dev-img", default=None)
    parser.add_argument("--dev-out-csv", default="outputs/apertus_dev_preds.csv")
    parser.add_argument("--out-dir", default="outputs/apertus-8b-lora")
    parser.add_argument("--hub-repo", default=None, help="Optional HF repo e.g. helshahaby/apertus-galaxus-lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    args, _ = parser.parse_known_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if torch.cuda.is_available() else 0
    use_4bit = vram_gb < 32.0
    print(f"Detected GPU VRAM: {vram_gb:.1f} GB | Using 4-bit QLoRA: {use_4bit}")

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["vision_tokenizer", "vqgan", "lm_head"],
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb_config, device_map={"": 0}, low_cpu_mem_usage=True, trust_remote_code=True
        )
        for name, module in model.named_modules():
            if "vision" in name or "vq" in name:
                module.to(torch.float32)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True
        )
        model.gradient_checkpointing_enable()

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
        push_to_hub=bool(args.hub_repo),
        hub_model_id=args.hub_repo,
        hub_private_repo=True,
    )
    Trainer(model=model, args=train_args, train_dataset=dataset, data_collator=collate_fn).train()
    model.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)
    if args.hub_repo:
        model.push_to_hub(args.hub_repo, private=True)
        processor.push_to_hub(args.hub_repo, private=True)

    if args.val_csv and args.val_img:
        val_preds = run_batched_inference(model, processor, args.val_csv, args.val_img, args.val_out_csv)
        val_truth = pd.read_csv(args.val_csv)
        if all(d in val_truth.columns for d in DIMENSIONS):
            evaluate_predictions(val_truth, val_preds)

    if args.dev_csv and args.dev_img:
        run_batched_inference(model, processor, args.dev_csv, args.dev_img, args.dev_out_csv)


if __name__ == "__main__":
    main()