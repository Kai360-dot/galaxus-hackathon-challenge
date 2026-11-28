"""LoRA on Apertus and the six heads, trained together, starting from the frozen-feature model.

The model is the one of heads.py: Apertus with the short prompt, the hidden state after layer 24 (last token and
the mean over the vision tokens), DINOv2 and pixel features appended, six MLP heads. Here Apertus gets a LoRA adapter
and learns together with the heads. The adapter starts at zero (LoRA's B matrices are zero), so at step 0 Apertus
computes exactly the frozen features, and the heads start from weights trained on those features: step 0 is the
frozen-feature model, and training moves on from there.

Speed: Apertus stops after layer 24 (the heads read nothing later), and a batch holds several images of different
sizes (left padding; checked to give the same features as one image at a time).

    # 1. heads on the frozen features, as the starting point (local, minutes)
    python src/lora_heads.py init --data-dir ~/galaxus/data --apertus results/heads_512/features \
        --extras results/dino_448_mirrored results/pixels_mirrored --output runs/lora_heads/init_heads.pt
    # 2. LoRA and heads together (GPU with bf16 Apertus), holdout score and a checkpoint every half epoch
    python src/lora_heads.py train --data-dir data --extras dino pixels --init runs/lora_heads/init_heads.pt \
        --output-dir runs/lora_heads --epochs 2
    # 3. probabilities for a part, with a checkpoint
    python src/lora_heads.py predict --data-dir data --extras dino pixels --checkpoint runs/lora_heads/step_1410 \
        --parts dev
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import common
from heads import (MIRRORED_PERSPECTIVES, SYSTEM_PROMPT, is_mirrorable, load_features, load_parts, make_head,
                   select, targets, train_heads, vision_token_mask)

LAYER = 24  # the heads read the hidden state after this many decoder layers
LORA_TARGETS = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(up|down)_proj)"


# ---------------------------------------------------------------- helper features (DINOv2, pixels)


def load_extras(directories, parts):
    """(image_id, mirrored) -> the helper features of all directories, concatenated, for the given parts."""
    table = {}
    for part in parts:
        merged = None
        for directory in directories:
            ids, flips, _, features, _ = load_features(Path(directory), part)
            flat = features.reshape(len(features), -1).astype(np.float32)
            current = {(str(i), bool(f)): v for i, f, v in zip(ids, flips, flat)}
            merged = current if merged is None else {k: np.concatenate([merged[k], current[k]]) for k in merged if k in current}
        table.update(merged)
    return table


# ---------------------------------------------------------------- 1. heads on the frozen features


def init(arguments):
    parts = load_parts(arguments.data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids, flips, _, features, layers = load_features(arguments.apertus, "train")
    x = select(features, [layers.index(LAYER)], "both")
    extras = load_extras(arguments.extras, ["train"])
    x = np.hstack([x, np.stack([extras[(str(i), bool(f))] for i, f in zip(ids, flips)])])
    y = targets(parts["train"], ids, flips)
    heads, mean, std = train_heads(x, y, arguments.hidden, arguments.epochs, device, arguments.seed)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"heads": {d: h.state_dict() for d, h in heads.items()}, "mean": mean.cpu(), "std": std.cpu(),
                "hidden": arguments.hidden, "layer": LAYER, "extras": [str(e) for e in arguments.extras]}, arguments.output)
    print(f"initial heads on {x.shape[1]} inputs saved to {arguments.output}")


# ---------------------------------------------------------------- the model: Apertus up to layer 24, LoRA, heads


class Heads(torch.nn.Module):
    def __init__(self, saved):
        super().__init__()
        inputs = saved["mean"].shape[1]
        self.register_buffer("mean", saved["mean"].float())
        self.register_buffer("std", saved["std"].float())
        self.heads = torch.nn.ModuleDict(
            {d: make_head(inputs, saved["hidden"], len(common.CLASSES[d])) for d in common.DIMENSIONS})
        state = saved["heads"]
        if all(isinstance(v, dict) for v in state.values()):  # {dimension: state_dict} from init
            state = {f"{d}.{k}": v for d, sd in state.items() for k, v in sd.items()}
        self.heads.load_state_dict(state)

    def forward(self, x):
        x = (x.float() - self.mean) / self.std
        return {d: head(x) for d, head in self.heads.items()}


def load_apertus(arguments, adapter=None, trainable=False):
    """Apertus cut after layer 24, with a LoRA adapter (new, or loaded), and a hook on the 24th layer's output."""
    from peft import LoraConfig, PeftModel, get_peft_model

    from zero_shot import load_model

    processor, model = load_model(arguments)
    language = model.model.language_model
    language.layers = torch.nn.ModuleList(list(language.layers)[:LAYER])
    if hasattr(model.config, "text_config"):
        model.config.text_config.num_hidden_layers = LAYER
    captured = {}
    language.layers[LAYER - 1].register_forward_hook(
        lambda module, inputs, output: captured.__setitem__("hidden", output[0] if isinstance(output, tuple) else output))
    if trainable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
    else:
        config = LoraConfig(r=arguments.lora_rank, lora_alpha=arguments.lora_alpha, lora_dropout=0.0,
                            target_modules=LORA_TARGETS, bias="none")
        model = get_peft_model(model, config)
    return processor, model, captured


def prompt_text(processor, row, image, with_context):
    user = common.user_prompt(row, with_context) + "\n\nClassify this image." if with_context else "Classify this image."
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)


def apertus_features(model, captured, batch):
    """Per image: the last-token state and the mean over its vision tokens, after layer 24 (2 x 4096)."""
    model(**batch, use_cache=False, logits_to_keep=1)
    hidden = captured.pop("hidden")
    vision = vision_token_mask(batch["input_ids"], model.config) & batch["attention_mask"].bool()
    mean = (hidden * vision.unsqueeze(-1)).sum(1) / vision.sum(1, keepdim=True).clamp(min=1)
    return torch.cat([hidden[:, -1], mean], dim=1)


# ---------------------------------------------------------------- data


class Images(torch.utils.data.Dataset):
    """Rows as model inputs: prepared image, prompt text, helper features, labels; augmentation for training."""

    def __init__(self, rows, processor, extras, image_size, train, no_context_share=1 / 3, seed=common.SEED):
        self.rows, self.processor, self.extras = rows.to_dict("records"), processor, extras
        self.image_size, self.train, self.no_context_share, self.seed = image_size, train, no_context_share, seed
        self.epoch, self.with_context = 0, True

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        rng = random.Random(hash((self.seed, self.epoch, index)))
        mirrored = self.train and is_mirrorable(row) and rng.random() < 0.5
        with_context = rng.random() >= self.no_context_share if self.train else self.with_context
        image = common.prepare_image(row["image_file"], self.image_size, mirrored=mirrored)
        labels = []
        for d in common.DIMENSIONS:
            label = row.get(d, "")
            if d == "perspective" and mirrored:
                label = MIRRORED_PERSPECTIVES.get(label, label)
            human = row.get(f"{d}_source") == "human" and label in common.CLASSES[d]
            labels.append(common.CLASSES[d].index(label) if human else -1)
        return {"text": prompt_text(self.processor, row, image, with_context), "image": image,
                "extras": self.extras[(str(row["image_id"]), mirrored)], "labels": labels, "image_id": row["image_id"]}


def collate(processor):
    def run(items):
        batch = processor(text=[i["text"] for i in items], images=[i["image"] for i in items], return_tensors="pt",
                          padding=True)
        return {"batch": dict(batch), "extras": torch.tensor(np.stack([i["extras"] for i in items])),
                "labels": torch.tensor([i["labels"] for i in items]), "image_ids": [i["image_id"] for i in items]}
    return run


def class_weights(rows, device):
    """count ** -0.5 per class of the human labels, normalized to a mean of 1 per label (as in heads.py)."""
    weights = {}
    for d in common.DIMENSIONS:
        human = rows.loc[rows[f"{d}_source"] == "human", d]
        counts = torch.tensor([float((human == c).sum()) for c in common.CLASSES[d]])
        w = (counts.clamp(min=1) ** -0.5) * (counts > 0)
        weights[d] = (w / (w * counts).sum() * counts.sum()).to(device)
    return weights


# ---------------------------------------------------------------- prediction and scoring


@torch.inference_mode()
def predict_rows(model, captured, heads, loader, device):
    """image_id -> dimension -> probabilities in common.CLASSES order."""
    model.eval(), heads.eval()
    result = {}
    for item in loader:
        batch = {k: v.to(device) for k, v in item["batch"].items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = apertus_features(model, captured, batch)
        logits = heads(torch.cat([features.float(), item["extras"].to(device)], dim=1))
        for k, image_id in enumerate(item["image_ids"]):
            result[str(image_id)] = {d: torch.softmax(logits[d][k].float(), 0).cpu().numpy() for d in common.DIMENSIONS}
    return result


def holdout_score(metric, rows, probabilities):
    answers = {i: {d: common.CLASSES[d][int(p[d].argmax())] for d in common.DIMENSIONS} for i, p in probabilities.items()}
    return common.score_rows(metric, rows, answers, {})


def loader_for(rows, processor, extras, arguments, train, with_context=True, shuffle=False, generator=None):
    dataset = Images(rows, processor, extras, arguments.image_size, train)
    dataset.with_context = with_context
    return dataset, torch.utils.data.DataLoader(dataset, batch_size=arguments.batch_size, shuffle=shuffle,
                                                num_workers=arguments.workers, collate_fn=collate(processor),
                                                generator=generator, persistent_workers=False)


# ---------------------------------------------------------------- 2. training


def save_checkpoint(model, heads, directory, upload):
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory / "adapter")
    torch.save({"heads": heads.heads.state_dict(), "mean": heads.mean.cpu(), "std": heads.std.cpu(),
                "hidden": heads.heads["subject"][1].out_features if len(heads.heads["subject"]) > 2 else 0,
                "extras": getattr(heads, "extras", None),
                "layer": LAYER}, directory / "heads.pt")
    if upload:
        upload(directory)


def train(arguments):
    parts = load_parts(arguments.data_dir)
    metric = common.load_metric(arguments.data_dir)
    rows = parts["train"] if arguments.limit == 0 else parts["train"].head(arguments.limit)
    holdout = parts["holdout"] if arguments.holdout_limit == 0 else parts["holdout"].head(arguments.holdout_limit)
    extras = load_extras(arguments.extras, ["train", "holdout"])
    processor, model, captured = load_apertus(arguments, trainable=True)
    device = next(model.parameters()).device
    heads = Heads(torch.load(arguments.init, map_location="cpu")).to(device)
    weights = class_weights(parts["train"], device)
    upload = make_uploader(arguments)

    dataset, _ = loader_for(rows, processor, extras, arguments, train=True)
    _, holdout_loader = loader_for(holdout, processor, extras, arguments, train=False)
    steps_per_epoch = math.ceil(len(dataset) / (arguments.batch_size * arguments.accumulation))
    total = max(1, round(steps_per_epoch * arguments.epochs))
    lora = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": lora, "lr": arguments.lora_lr},
                                   {"params": heads.parameters(), "lr": arguments.heads_lr}], weight_decay=0.01)
    warmup = max(1, round(0.03 * total))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    evaluate_every = max(1, steps_per_epoch // arguments.evaluations_per_epoch)
    print(f"{len(dataset)} train images, batch {arguments.batch_size} x {arguments.accumulation}, {total} steps, "
          f"evaluation every {evaluate_every} steps", flush=True)

    def evaluate(step):
        started = time.time()
        metrics = holdout_score(metric, holdout, predict_rows(model, captured, heads, holdout_loader, device))
        print(f"== step {step}: holdout macro F1 {metrics['overall']['f1_macro']:.4f} | weighted "
              f"{metrics['overall']['f1_weighted']:.4f} | "
              + " | ".join(f"{d[:8]} {metrics[d]['f1_macro']:.3f}" for d in common.DIMENSIONS if d in metrics)
              + f" | {len(holdout) / (time.time() - started):.1f} images/s", flush=True)
        model.train(), heads.train()

    if arguments.evaluate_start:
        evaluate(0)
    model.train(), heads.train()
    step, seen, losses, started, epoch = 0, 0, [], time.time(), 0
    while step < total:
        dataset.epoch = epoch
        loader = torch.utils.data.DataLoader(dataset, batch_size=arguments.batch_size, shuffle=True,
                                             num_workers=arguments.workers, collate_fn=collate(processor),
                                             generator=torch.Generator().manual_seed(arguments.seed + epoch))
        try:
            for k, item in enumerate(loader):
                batch = {key: v.to(device) for key, v in item["batch"].items()}
                labels = item["labels"].to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    features = apertus_features(model, captured, batch)
                logits = heads(torch.cat([features.float(), item["extras"].to(device)], dim=1))
                loss = sum(torch.nn.functional.cross_entropy(logits[d].float(), labels[:, j], weight=weights[d],
                                                             ignore_index=-1)
                           for j, d in enumerate(common.DIMENSIONS) if (labels[:, j] >= 0).any()) / len(common.DIMENSIONS)
                (loss / arguments.accumulation).backward()
                losses.append(loss.item())
                seen += len(item["image_ids"])
                if (k + 1) % arguments.accumulation:
                    continue
                torch.nn.utils.clip_grad_norm_(lora + list(heads.parameters()), 1.0)
                optimizer.step(), schedule.step(), optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % arguments.log_every == 0:
                    rate = seen / (time.time() - started)
                    print(f"step {step}/{total} | loss {np.mean(losses[-50:]):.4f} | {rate:.2f} images/s | "
                          f"eta {(total - step) * arguments.batch_size * arguments.accumulation / rate / 60:.0f} min | "
                          f"peak {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB", flush=True)
                if step % evaluate_every == 0 or step == total:
                    evaluate(step)
                    save_checkpoint(model, heads, arguments.output_dir / f"step_{step}", upload)
                if step >= total:
                    break
        except torch.cuda.OutOfMemoryError:
            # A batch too large for this GPU (images differ in size): halve the batch, keep the effective batch by
            # accumulating, and go on with this epoch.
            if arguments.batch_size == 1:
                raise
            optimizer.zero_grad(set_to_none=True)
            captured.clear()
            torch.cuda.empty_cache()
            arguments.batch_size //= 2
            arguments.accumulation *= 2
            print(f"out of memory: now batch {arguments.batch_size} x {arguments.accumulation}", flush=True)
            continue
        epoch += 1
    print(f"done after {(time.time() - started) / 60:.0f} min", flush=True)


def make_uploader(arguments):
    """Upload a checkpoint folder to the private results repo, so a stopped job keeps what it saved."""
    if not arguments.results_repo:
        return None
    from huggingface_hub import HfApi

    api = HfApi()

    def upload(directory):
        try:
            api.upload_folder(repo_id=arguments.results_repo, repo_type="dataset", folder_path=str(directory),
                              path_in_repo=f"{arguments.run_name}/{directory.name}")
            print(f"uploaded {directory.name}", flush=True)
        except Exception as error:  # a failed upload must not stop the training
            print(f"upload of {directory.name} failed: {error}", flush=True)
    return upload


# ---------------------------------------------------------------- 3. prediction


def predict(arguments):
    parts = load_parts(arguments.data_dir)
    metric = common.load_metric(arguments.data_dir)
    extras = load_extras(arguments.extras, arguments.parts)
    processor, model, captured = load_apertus(arguments, adapter=arguments.checkpoint / "adapter")
    device = next(model.parameters()).device
    saved = torch.load(arguments.checkpoint / "heads.pt", map_location="cpu")
    heads = Heads(saved).to(device)
    out = arguments.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for part in arguments.parts:
        rows = parts[part] if arguments.holdout_limit == 0 else parts[part].head(arguments.holdout_limit)
        for with_context in ((True, False) if part == "holdout" else (True,)):
            _, loader = loader_for(rows, processor, extras, arguments, train=False, with_context=with_context)
            started = time.time()
            probabilities = predict_rows(model, captured, heads, loader, device)
            name = f"{part}_{'context' if with_context else 'no_context'}"
            np.save(out / f"probabilities_{name}.npy", probabilities, allow_pickle=True)
            print(f"{name}: {len(rows)} images, {len(rows) / (time.time() - started):.1f} images/s", flush=True)
            if part == "holdout":
                common.print_metrics(holdout_score(metric, rows, probabilities))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["init", "train", "predict"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--apertus", type=Path, help="init: frozen Apertus features (heads.py extract)")
    parser.add_argument("--extras", type=Path, nargs="+", required=True, help="DINOv2 and pixel feature folders")
    parser.add_argument("--output", type=Path, default=Path("runs/lora_heads/init_heads.pt"))
    parser.add_argument("--init", type=Path, help="train: the initial heads of `init`")
    parser.add_argument("--checkpoint", type=Path, help="predict: a step_* folder")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/lora_heads"))
    parser.add_argument("--parts", nargs="+", default=["dev"])
    parser.add_argument("--model", default="swiss-ai/Apertus-v1.5-8B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="none")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--heads-lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="train images, 0 for all (tests)")
    parser.add_argument("--holdout-limit", type=int, default=0, help="holdout images, 0 for all (tests)")
    parser.add_argument("--no-evaluate-start", dest="evaluate_start", action="store_false")
    parser.add_argument("--evaluations-per-epoch", type=int, default=4, help="holdout score and checkpoint this often")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--results-repo", help="private dataset repo for checkpoints")
    parser.add_argument("--run-name", default="lora_heads")
    parser.add_argument("--seed", type=int, default=common.SEED)
    arguments = parser.parse_args()
    if arguments.command == "init":
        arguments.epochs = int(arguments.epochs) if arguments.epochs > 2 else 30
    {"init": init, "train": train, "predict": predict}[arguments.command](arguments)


if __name__ == "__main__":
    main()
