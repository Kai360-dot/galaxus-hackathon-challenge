"""LoRA on Apertus and six classification heads, trained together from scratch within a time budget.

Apertus 1.5 turns an image into discrete tokens and reads them with its language layers. So these layers are where
it can learn to see what the labels ask for. Here they get a LoRA adapter, and six small heads read the hidden state
after layer 24: the last token, the mean over the vision tokens, and the differences of that mean between the left
and the right half and between the top and the bottom half of the image. A plain mean cannot tell left from right,
which the perspective needs. Adapter and heads both start untrained and learn together. Unlike lora_heads.py, the
heads do not start from weights fitted on frozen features, and no helper features are needed: the Kaggle files are
enough.

The vision tokenizer runs in float32 and outside the mixed-precision block: its codes are an argmax over 131,072
entries and the model source asks for float32 there.

Training and prediction take one forward pass per image, in batches of images of different sizes. The learning rate
follows a cosine over the time budget instead of a step count, so a run ends when planned on any GPU.

    python src/lora_classifier.py train --data-dir data --output-dir runs/lora_classifier --minutes 100
    python src/lora_classifier.py predict --data-dir data --checkpoint runs/lora_classifier/checkpoint \
        --output-dir runs/lora_classifier/final --parts dev --csv data/final/final.csv

train scores the holdout every few minutes, saves a checkpoint each time and ends with holdout and dev predictions.
predict writes, per part, the class probabilities in the format that ensemble.py reads with --probs, and one
submission file with every predicted row.
"""

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

import common
from heads import MIRRORED_PERSPECTIVES, is_mirrorable, load_parts
from heads import vision_token_mask
from lora_heads import LORA_TARGETS, class_weights, make_uploader, prompt_text

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Rows of rare classes are shown more often per epoch, as in the reference notebook.
RARE_CLASS_TARGET = 200
MAX_REPEATS = 3
WARMUP_SHARE = 0.04  # of the time budget
# A repeated row is prepared at another size than the first copy, so the adapter does not see the same tokens again.
REPEAT_SIZE_FACTORS = (0.875, 0.9375, 1.0625)
STATES_PER_IMAGE = 4  # last token, vision mean, left minus right, top minus bottom


# ---------------------------------------------------------------- the model: Apertus cut short, LoRA, heads


def load_apertus(arguments, adapter=None):
    """Apertus cut after arguments.layers decoder layers, with a LoRA adapter (new, or loaded) and a hook that
    captures the output of the last kept layer."""
    from peft import LoraConfig, PeftModel, get_peft_model

    from zero_shot import load_model

    processor, model = load_model(arguments)
    processor.tokenizer.padding_side = "left"  # so the last position of every row is its last prompt token
    language = model.model.language_model
    language.layers = torch.nn.ModuleList(list(language.layers)[: arguments.layers])
    if hasattr(model.config, "text_config"):
        model.config.text_config.num_hidden_layers = arguments.layers
    captured = {}
    language.layers[-1].register_forward_hook(
        lambda module, inputs, output: captured.__setitem__("hidden", output[0] if isinstance(output, tuple) else output))
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter))
    else:
        config = LoraConfig(r=arguments.lora_rank, lora_alpha=arguments.lora_alpha, lora_dropout=0.0,
                            target_modules=LORA_TARGETS, bias="none")
        model = get_peft_model(model, config)
        for parameter in model.parameters():
            if parameter.requires_grad:  # the adapter learns in float32, a bf16 weight loses small updates
                parameter.data = parameter.data.float()
    # Always in eval mode: the adapter has no dropout, and nothing else in Apertus should change with the mode.
    model.eval()
    return processor, model, captured


class Heads(torch.nn.Module):
    """Six small MLPs on the standardized features. Mean and spread are measured once, on the untrained model."""

    def __init__(self, inputs, hidden, dropout=0.1):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, inputs))
        self.register_buffer("std", torch.ones(1, inputs))
        self.heads = torch.nn.ModuleDict({
            d: torch.nn.Sequential(torch.nn.Linear(inputs, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout),
                                   torch.nn.Linear(hidden, len(common.CLASSES[d])))
            for d in common.DIMENSIONS})

    def forward(self, features):
        x = ((features.float() - self.mean) / self.std).clamp(-10, 10)
        return {d: head(x) for d, head in self.heads.items()}


# ---------------------------------------------------------------- data


def repeat_counts(rows):
    """How often each row is shown per epoch: enough to lift its rarest class towards RARE_CLASS_TARGET."""
    records = rows.to_dict("records")
    human = [[d for d in common.DIMENSIONS if row[f"{d}_source"] == "human"] for row in records]
    counts = Counter((d, row[d]) for row, dimensions in zip(records, human) for d in dimensions)
    repeats = []
    for row, dimensions in zip(records, human):
        rarest = min((counts[(d, row[d])] for d in dimensions), default=RARE_CLASS_TARGET)
        repeats.append(min(MAX_REPEATS, max(1, math.ceil(RARE_CLASS_TARGET / rarest))))
    return repeats


class Images(torch.utils.data.Dataset):
    """Rows as model inputs: prepared image, prompt text and labels, -1 where there is no human label.

    With repeats (training) a row appears that many times per epoch, every further copy at a slightly different
    size. Mirrorable images are mirrored half of the time with left and right swapped in the perspective, and a
    share of the rows loses its product context.
    """

    def __init__(self, rows, processor, image_size, repeats=None, no_context_share=0.0, seed=common.SEED):
        records = rows.to_dict("records")
        self.train = repeats is not None
        self.items = [(row, copy) for row, count in zip(records, repeats or [1] * len(records)) for copy in range(count)]
        self.processor, self.image_size = processor, image_size
        self.no_context_share, self.seed, self.epoch = no_context_share, seed, 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        row, copy = self.items[index]
        rng = random.Random(hash((self.seed, self.epoch, index)))
        mirrored = self.train and is_mirrorable(row) and rng.random() < 0.5
        with_context = not self.train or rng.random() >= self.no_context_share
        size = self.image_size
        if copy:
            size = common.PATCH_SIZE * round(size * rng.choice(REPEAT_SIZE_FACTORS) / common.PATCH_SIZE)
        try:
            image = common.prepare_image(row["image_file"], size, mirrored=mirrored)
        except Exception as error:  # an unreadable file must not stop a run over thousands of images
            print(f"{row['image_id']}: image not readable ({error}), a white image is used", flush=True)
            image = Image.new("RGB", (256, 256), "white")
        labels = []
        for d in common.DIMENSIONS:
            label = row.get(d, "")
            if d == "perspective" and mirrored:
                label = MIRRORED_PERSPECTIVES.get(label, label)
            human = row.get(f"{d}_source") == "human" and label in common.CLASSES[d]
            labels.append(common.CLASSES[d].index(label) if human else -1)
        return {"text": prompt_text(self.processor, row, image, with_context), "image": image, "labels": labels,
                "image_id": row["image_id"]}


class Collate:
    """A batch of images of different sizes as one padded model input."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, items):
        batch = self.processor(text=[i["text"] for i in items], images=[i["image"] for i in items],
                               return_tensors="pt", padding=True)
        return {"batch": dict(batch), "labels": torch.tensor([i["labels"] for i in items]),
                "image_ids": [i["image_id"] for i in items]}


def loader_for(dataset, processor, batch_size, workers, shuffle_seed=None):
    generator = None if shuffle_seed is None else torch.Generator().manual_seed(shuffle_seed)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_seed is not None,
                                       num_workers=workers, collate_fn=Collate(processor), generator=generator)


def features_of(model, captured, batch, device):
    """Per image four pooled states after the last kept layer: the last token, the mean over the vision tokens,
    and that mean on the left half minus the right half and on the top half minus the bottom half."""
    input_ids, attention = batch["input_ids"].to(device), batch["attention_mask"].to(device)
    sizes = batch["image_sizes"].to(device)
    apertus = model.get_base_model() if hasattr(model, "get_base_model") else model
    with torch.no_grad():  # float32 and outside the autocast below: the codes are an argmax over a large codebook
        codes = apertus.model.get_image_tokens(batch["pixel_values"].to(device), sizes)
    placeholders = input_ids == apertus.config.image_token_id
    assert int(placeholders.sum()) == codes.numel(), "image placeholders and image codes differ in number"
    input_ids = input_ids.masked_scatter(placeholders, codes)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        model(input_ids=input_ids, attention_mask=attention, use_cache=False, logits_to_keep=1)
    hidden = captured.pop("hidden").float()

    # The vision tokens of an image come row by row: token k sits in row k // width and column k % width.
    vision = vision_token_mask(input_ids, apertus.config) & attention.bool()
    index = vision.long().cumsum(1) - 1
    height, width = (sizes[:, 0:1] // common.PATCH_SIZE), (sizes[:, 1:2] // common.PATCH_SIZE)
    row, column = torch.div(index, width, rounding_mode="floor"), index % width
    halves = [column < width // 2, column >= width - width // 2, row < height // 2, row >= height - height // 2]
    masks = torch.stack([vision] + [vision & half for half in halves], dim=1).float()
    means = torch.bmm(masks, hidden) / masks.sum(2, keepdim=True).clamp(min=1)
    return torch.cat([hidden[:, -1], means[:, 0], means[:, 1] - means[:, 2], means[:, 3] - means[:, 4]], dim=1)


# ---------------------------------------------------------------- prediction and scoring


@torch.inference_mode()
def predict_rows(model, captured, heads, processor, rows, arguments, device):
    """image_id -> dimension -> probabilities in common.CLASSES order, with the product context."""
    heads.eval()
    dataset = Images(rows, processor, arguments.image_size)
    result = {}
    for item in loader_for(dataset, processor, arguments.predict_batch_size, arguments.workers):
        logits = heads(features_of(model, captured, item["batch"], device))
        for k, image_id in enumerate(item["image_ids"]):
            result[str(image_id)] = {d: torch.softmax(logits[d][k].float(), 0).cpu().numpy() for d in common.DIMENSIONS}
    heads.train()
    return result


def answers_of(probabilities):
    return {image_id: {d: common.CLASSES[d][int(p[d].argmax())] for d in common.DIMENSIONS}
            for image_id, p in probabilities.items()}


def write_probabilities(path, probabilities):
    """One line per image: {"image_id", "probabilities": {dimension: {class: p}}}, as ensemble.py reads it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        for image_id, p in probabilities.items():
            record = {d: {name: round(float(value), 6) for name, value in zip(common.CLASSES[d], p[d])}
                      for d in common.DIMENSIONS}
            output.write(json.dumps({"image_id": image_id, "probabilities": record}) + "\n")


def write_submission(data_dir, path, answers):
    """sample_submission.csv with every predicted row filled in; the other rows keep their constants."""
    sample = pd.read_csv(data_dir / "sample_submission.csv", dtype=str, keep_default_na=False)
    submission = sample.set_index("image_id")
    predicted = [image_id for image_id in submission.index if image_id in answers]
    submission.loc[predicted, list(common.DIMENSIONS)] = [[answers[i][d] for d in common.DIMENSIONS] for i in predicted]
    for d in common.DIMENSIONS:
        assert submission[d].isin(common.CLASSES[d]).all(), f"{d} has an invalid cell"
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.reset_index().to_csv(path, index=False)
    print(f"{path}: {len(predicted)} of {len(submission)} rows predicted", flush=True)


def predict_parts(model, captured, heads, processor, named_rows, metric, arguments, device, upload):
    """Probabilities per part, the holdout score, and one submission file with all predicted rows."""
    directory = arguments.output_dir / "predictions"
    answers = {}
    for name, rows in named_rows.items():
        if arguments.predict_limit:
            rows = rows.head(arguments.predict_limit)
        started = time.time()
        probabilities = predict_rows(model, captured, heads, processor, rows, arguments, device)
        print(f"\n===== {name}: {len(rows)} images, {len(rows) / (time.time() - started):.1f} images/s, "
              f"peak {peak_memory():.1f} GB", flush=True)
        write_probabilities(directory / f"{name}_context.jsonl", probabilities)
        if name == "holdout":
            common.print_metrics(common.score_rows(metric, rows, answers_of(probabilities), {}))
        else:
            answers.update(answers_of(probabilities))
    if answers:
        write_submission(arguments.data_dir, directory / "submission.csv", answers)
    if upload:
        upload(directory)


def peak_memory():
    return torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0


# ---------------------------------------------------------------- training


@torch.inference_mode()
def feature_statistics(model, captured, processor, rows, arguments, device):
    """Mean and spread of every feature over a sample of train images, on the untrained model."""
    sample = rows.sample(min(len(rows), arguments.calibration_images), random_state=arguments.seed)
    dataset = Images(sample, processor, arguments.image_size)
    chunks = [features_of(model, captured, item["batch"], device).float().cpu()
              for item in loader_for(dataset, processor, arguments.predict_batch_size, arguments.workers)]
    features = torch.cat(chunks)
    return features.mean(0, keepdim=True), features.std(0, keepdim=True).clamp(min=1e-3)


def rate_factor(fraction):
    """Learning rate as a share of its peak: a linear warmup, then a cosine down to zero at the end of the budget."""
    return min(1.0, fraction / WARMUP_SHARE) * 0.5 * (1 + math.cos(math.pi * min(1.0, fraction)))


def save_checkpoint(model, heads, arguments, upload):
    directory = arguments.output_dir / "checkpoint"
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory / "adapter")
    torch.save({"heads": heads.state_dict(), "hidden": arguments.hidden, "layers": arguments.layers,
                "image_size": arguments.image_size, "model": arguments.model, "revision": arguments.revision},
               directory / "heads.pt")
    if upload:
        upload(directory)


def train(arguments):
    parts = load_parts(arguments.data_dir)
    metric = common.load_metric(arguments.data_dir)
    rows = parts["train"].head(arguments.limit) if arguments.limit else parts["train"]
    holdout = parts["holdout"].head(arguments.holdout_limit) if arguments.holdout_limit else parts["holdout"]
    processor, model, captured = load_apertus(arguments)
    device = next(model.parameters()).device
    upload = make_uploader(arguments)

    heads = Heads(STATES_PER_IMAGE * model.config.text_config.hidden_size, arguments.hidden).to(device)
    mean, std = feature_statistics(model, captured, processor, rows, arguments, device)
    heads.mean.copy_(mean), heads.std.copy_(std)
    weights = class_weights(parts["train"], device)
    lora = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": lora, "lr": arguments.lora_lr},
                                   {"params": heads.parameters(), "lr": arguments.heads_lr}], weight_decay=0.01)
    peak_rates = [arguments.lora_lr, arguments.heads_lr]

    dataset = Images(rows, processor, arguments.image_size, repeat_counts(rows), arguments.no_context_share,
                     arguments.seed)
    batch_size, accumulation = arguments.batch_size, arguments.accumulation
    budget = arguments.minutes * 60
    print(f"{len(rows)} train images, {len(dataset)} rows per epoch with repeats | batch {batch_size} x {accumulation} | "
          f"{sum(p.numel() for p in lora) / 1e6:.0f} M adapter weights | {arguments.minutes:.0f} minutes", flush=True)

    def evaluate(step):
        started = time.time()
        metrics = common.score_rows(
            metric, holdout, answers_of(predict_rows(model, captured, heads, processor, holdout, arguments, device)), {})
        print(f"== step {step}: holdout macro F1 {metrics['overall']['f1_macro']:.4f} | weighted "
              f"{metrics['overall']['f1_weighted']:.4f} | "
              + " | ".join(f"{d[:8]} {metrics[d]['f1_macro']:.3f}" for d in common.DIMENSIONS if d in metrics)
              + f" | {len(holdout) / (time.time() - started):.1f} images/s", flush=True)

    heads.train()
    started = time.time()
    step, seen, epoch, pending, losses, finished = 0, 0, 0, 0, [], False
    next_evaluation = arguments.evaluate_minutes * 60
    while not finished:
        dataset.epoch = epoch
        loader = loader_for(dataset, processor, batch_size, arguments.workers, shuffle_seed=arguments.seed + epoch)
        features = logits = loss = None
        try:
            for item in loader:
                labels = item["labels"].to(device)
                features = features_of(model, captured, item["batch"], device)
                logits = heads(features)
                terms = [torch.nn.functional.cross_entropy(logits[d], labels[:, j], weight=weights[d], ignore_index=-1)
                         for j, d in enumerate(common.DIMENSIONS) if (labels[:, j] >= 0).any()]
                if not terms:  # no human label in the whole batch
                    continue
                loss = sum(terms) / len(common.DIMENSIONS)
                (loss / accumulation).backward()
                losses.append(loss.item())
                seen += len(item["image_ids"])
                pending += 1
                if pending < accumulation:
                    continue
                elapsed = time.time() - started
                for group, peak in zip(optimizer.param_groups, peak_rates):
                    group["lr"] = peak * rate_factor(elapsed / budget)
                torch.nn.utils.clip_grad_norm_(lora + list(heads.parameters()), 1.0)
                optimizer.step(), optimizer.zero_grad(set_to_none=True)
                step, pending = step + 1, 0
                if step % arguments.log_every == 0:
                    print(f"step {step} | epoch {seen / len(dataset):.2f} | loss {sum(losses[-50:]) / len(losses[-50:]):.4f} | "
                          f"{seen / elapsed:.2f} images/s | {elapsed / 60:.0f} of {arguments.minutes:.0f} min | "
                          f"peak {peak_memory():.1f} GB", flush=True)
                if elapsed >= budget:
                    finished = True
                    break
                if elapsed >= next_evaluation:
                    evaluate(step)
                    save_checkpoint(model, heads, arguments, upload)
                    next_evaluation += arguments.evaluate_minutes * 60
        except torch.cuda.OutOfMemoryError:
            # A batch too large for this GPU: halve the batch, keep the effective batch by accumulating, go on.
            if batch_size == 1:
                raise
            features = logits = loss = None
            optimizer.zero_grad(set_to_none=True)
            captured.clear()
            torch.cuda.empty_cache()
            batch_size, accumulation, pending = batch_size // 2, accumulation * 2, 0
            print(f"out of memory: now batch {batch_size} x {accumulation}", flush=True)
        epoch += 1

    print(f"training done: {step} steps, {seen / len(dataset):.2f} epochs in {(time.time() - started) / 60:.0f} min",
          flush=True)
    save_checkpoint(model, heads, arguments, upload)
    named_rows = {"holdout": holdout, **{part: parts[part] for part in arguments.parts if part != "holdout"}}
    predict_parts(model, captured, heads, processor, named_rows, metric, arguments, device, upload)


# ---------------------------------------------------------------- prediction with a checkpoint


def predict(arguments):
    saved = torch.load(arguments.checkpoint / "heads.pt", map_location="cpu")
    arguments.layers, arguments.hidden, arguments.image_size = saved["layers"], saved["hidden"], saved["image_size"]
    parts = load_parts(arguments.data_dir)
    metric = common.load_metric(arguments.data_dir)
    processor, model, captured = load_apertus(arguments, adapter=arguments.checkpoint / "adapter")
    device = next(model.parameters()).device
    heads = Heads(saved["heads"]["mean"].shape[1], saved["hidden"])
    heads.load_state_dict(saved["heads"])
    heads.to(device)
    named_rows = {part: parts[part] for part in arguments.parts}
    for csv_path in arguments.csv:
        named_rows[csv_path.stem] = common.read_part(csv_path)
    predict_parts(model, captured, heads, processor, named_rows, metric, arguments, device, make_uploader(arguments))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/lora_classifier"))
    parser.add_argument("--checkpoint", type=Path, help="predict: the checkpoint folder of a training run")
    parser.add_argument("--parts", nargs="+", default=["dev"], help="parts to predict; train always adds the holdout")
    parser.add_argument("--csv", type=Path, nargs="*", default=[], help="predict: more csv files, such as final.csv")
    parser.add_argument("--model", default="swiss-ai/Apertus-v1.5-8B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="none")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--layers", type=int, default=24, help="decoder layers of Apertus that are run")
    parser.add_argument("--hidden", type=int, default=512, help="width of the hidden layer of each head")
    parser.add_argument("--minutes", type=float, default=100.0, help="train: time budget, evaluations included")
    parser.add_argument("--evaluate-minutes", type=float, default=20.0, help="holdout score and checkpoint this often")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=2, help="batches per optimizer step")
    parser.add_argument("--predict-batch-size", type=int, default=32)
    parser.add_argument("--lora-lr", type=float, default=2e-4)
    parser.add_argument("--heads-lr", type=float, default=5e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--no-context-share", type=float, default=0.2, help="train rows shown without product context")
    parser.add_argument("--calibration-images", type=int, default=384)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0, help="train images, 0 for all (tests)")
    parser.add_argument("--holdout-limit", type=int, default=0, help="holdout images, 0 for all (tests)")
    parser.add_argument("--predict-limit", type=int, default=0, help="images per predicted part, 0 for all (tests)")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--results-repo", help="private dataset repo for checkpoints and predictions")
    parser.add_argument("--run-name", default="lora_classifier")
    parser.add_argument("--seed", type=int, default=common.SEED)
    arguments = parser.parse_args()
    {"train": train, "predict": predict}[arguments.command](arguments)


if __name__ == "__main__":
    main()
