"""LoRA finetune of Apertus on the atomic questions, asked one after the other in one conversation.

One training sequence per image:

    [system] [image] [context]  Q1 subject ... Answer: A  Q2 background ... Answer: B  ...  Q6 text_language ... Answer: G

The loss sits only on the six answer letters, as a cross-entropy over the option letters of that question (class
weighted, human labels only). Prediction runs the same conversation: the image once, then question after question,
each time reading the letter probabilities and appending the most likely letter before the next question.

Augmentation: mirrored images (left and right swapped in the perspective label) for left/right perspectives without
text, and context dropout (the product context left out for a share of the images).

    python src/lora_atomic.py train --data-dir ~/galaxus/data --output-dir runs/lora --epochs 1
    python src/lora_atomic.py predict --data-dir ~/galaxus/data --adapter runs/lora/adapter --parts holdout dev
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
from tqdm.auto import tqdm

import common
from atomic import QUESTIONS, SYSTEM_PROMPT, question_text
from heads import MIRRORED_PERSPECTIVES, is_mirrorable, load_parts

LORA_TARGETS = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(up|down)_proj)"
MARKER = "<<SPLIT>>"
ANSWER_PREFIX = "Answer:"


# ---------------------------------------------------------------- the conversation as token pieces


class Conversation:
    """Token pieces of the atomic conversation, the same in training and prediction."""

    def __init__(self, processor):
        self.processor = processor
        tokenizer = processor.tokenizer
        # The text between the user's question and the answer, and between an answer and the next question.
        user_turn = [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": MARKER}]}]
        text = processor.apply_chat_template(user_turn, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        self.prefix_template, self.to_answer = text.split(MARKER)
        two_turns = [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": "QQQ"}, {"role": "assistant", "content": "AAA"},
                     {"role": "user", "content": "RRR"}]
        rendered = processor.apply_chat_template(two_turns, tokenize=False, enable_thinking=False)
        self.between = rendered[rendered.index("AAA") + 3 : rendered.index("RRR")]
        self.question_ids = {
            d: tokenizer(question_text(d) + self.to_answer + ANSWER_PREFIX, add_special_tokens=False)["input_ids"]
            for d in common.DIMENSIONS
        }
        self.between_ids = tokenizer(self.between, add_special_tokens=False)["input_ids"]
        # The letter of each option as it follows "Answer:" ( " A"), one token each.
        self.letter_ids = {}
        for d, (_, options) in QUESTIONS.items():
            ids = [tokenizer.encode(" " + chr(65 + i), add_special_tokens=False) for i in range(len(options))]
            assert all(len(i) == 1 for i in ids), f"option letters of {d} are not single tokens: {ids}"
            self.letter_ids[d] = [i[0] for i in ids]

    def prefix(self, image, user_text):
        """Tokens and pixels of system, image and context; the first question follows."""
        text = self.prefix_template.replace(MARKER, "")
        # The question text goes after the user's own text, inside the same user turn.
        content_text = (user_text + "\n\n") if user_text else ""
        text = text + content_text
        return self.processor(text=[text], images=[image], return_tensors="pt")

    def training_sequence(self, image, user_text, answers):
        """input_ids, pixels and per dimension the position whose next-token logits must give its answer letter."""
        prefix = self.prefix(image, user_text)
        ids = prefix["input_ids"][0].tolist()
        positions = {}
        for index, d in enumerate(common.DIMENSIONS):
            ids += self.question_ids[d]
            positions[d] = len(ids) - 1
            ids.append(self.letter_ids[d][answers[d]])
            if index < len(common.DIMENSIONS) - 1:
                ids += self.between_ids
        return prefix, torch.tensor([ids]), positions


# ---------------------------------------------------------------- data


def class_weights(rows):
    """Per dimension a weight per class: rare classes weigh more (count ** -0.5), normalized to mean 1 per label."""
    weights = {}
    for d in common.DIMENSIONS:
        human = rows.loc[rows[f"{d}_source"] == "human", d]
        counts = np.array([float((human == name).sum()) for name in common.CLASSES[d]])
        w = np.where(counts > 0, np.maximum(counts, 1) ** -0.5, 0.0)
        w = w / (w * counts).sum() * counts.sum()
        weights[d] = torch.tensor(w, dtype=torch.float32)
    return weights


class TrainingImages(torch.utils.data.Dataset):
    def __init__(self, rows, conversation, image_size, no_context_share, mirror, seed):
        self.rows = rows.to_dict("records")
        self.conversation, self.image_size = conversation, image_size
        self.no_context_share, self.mirror, self.seed = no_context_share, mirror, seed
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        rng = random.Random(hash((self.seed, self.epoch, index)))
        row = self.rows[index]
        mirrored = self.mirror and is_mirrorable(row) and rng.random() < 0.5
        with_context = rng.random() >= self.no_context_share
        image = common.prepare_image(row["image_file"], self.image_size, mirrored=mirrored)
        answers, trained = {}, {}
        for d in common.DIMENSIONS:
            label = row[d]
            if d == "perspective" and mirrored:
                label = MIRRORED_PERSPECTIVES[label]
            answers[d] = common.CLASSES[d].index(label)
            trained[d] = row[f"{d}_source"] == "human"
        user_text = common.user_prompt(row, with_context) if with_context else ""
        prefix, input_ids, positions = self.conversation.training_sequence(image, user_text, answers)
        return {"prefix": dict(prefix), "input_ids": input_ids, "positions": positions, "answers": answers,
                "trained": trained}


# ---------------------------------------------------------------- model


def load_for_training(arguments):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    from zero_shot import load_model

    processor, model = load_model(arguments)
    if arguments.quantization != "none":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    config = LoraConfig(r=arguments.lora_rank, lora_alpha=arguments.lora_alpha, lora_dropout=0.0,
                        target_modules=LORA_TARGETS, bias="none")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return processor, model


def answer_logits(model, sample, device):
    """Logits over the option letters at the six answer positions, from one forward pass."""
    prefix = {k: v.to(device) for k, v in sample["prefix"].items() if k != "input_ids" and k != "attention_mask"}
    input_ids = sample["input_ids"].to(device)
    positions = torch.tensor([sample["positions"][d] for d in common.DIMENSIONS], device=device)
    output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), logits_to_keep=positions,
                   use_cache=False, **prefix)
    logits = output.logits[0]
    assert logits.shape[0] == len(common.DIMENSIONS), f"expected 6 logit rows, got {tuple(logits.shape)}"
    return logits


def train(arguments):
    parts = load_parts(arguments.data_dir)
    rows = parts["train"] if arguments.limit == 0 else parts["train"].head(arguments.limit)
    processor, model = load_for_training(arguments)
    device = next(model.parameters()).device
    conversation = Conversation(processor)
    weights = {d: w.to(device) for d, w in class_weights(parts["train"]).items()}
    letter_ids = {d: torch.tensor(ids, device=device) for d, ids in conversation.letter_ids.items()}

    dataset = TrainingImages(rows, conversation, arguments.image_size, arguments.no_context_share,
                             arguments.mirror, arguments.seed)
    steps_per_epoch = math.ceil(len(dataset) / arguments.accumulation)
    total_steps = max(1, round(steps_per_epoch * arguments.epochs))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=arguments.learning_rate, weight_decay=0.01)
    warmup = max(1, round(0.05 * total_steps))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total_steps)))
    )
    print(f"{len(dataset)} images, {total_steps} optimizer steps of {arguments.accumulation} images "
          f"({arguments.epochs} epochs)", flush=True)

    output_dir = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    step, seen, running, started = 0, 0, [], time.time()
    epoch = 0
    while step < total_steps:
        dataset.epoch = epoch
        loader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=True, num_workers=arguments.workers,
                                             generator=torch.Generator().manual_seed(arguments.seed + epoch))
        for sample in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = answer_logits(model, sample, device)
            loss, count = 0.0, 0
            for row, d in enumerate(common.DIMENSIONS):
                if not sample["trained"][d]:
                    continue
                option_logits = logits[row, letter_ids[d]].float()
                target = torch.tensor(sample["answers"][d], device=device)
                loss = loss + weights[d][target] * torch.nn.functional.cross_entropy(option_logits, target)
                count += 1
            if count:
                (loss / count / arguments.accumulation).backward()
                running.append(loss.item() / count)
            seen += 1
            if seen % arguments.accumulation == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % arguments.log_every == 0 or step == total_steps:
                    elapsed = time.time() - started
                    print(f"step {step}/{total_steps} | loss {np.mean(running):.4f} | {seen / elapsed:.2f} images/s | "
                          f"lr {schedule.get_last_lr()[0]:.2e} | eta {(total_steps - step) * arguments.accumulation / (seen / elapsed) / 60:.0f} min | "
                          f"peak {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB", flush=True)
                    running = []
                if arguments.save_every and step % arguments.save_every == 0:
                    model.save_pretrained(output_dir / "adapter")
                if step >= total_steps:
                    break
        epoch += 1
    model.save_pretrained(output_dir / "adapter")
    print(f"adapter saved to {output_dir / 'adapter'} after {(time.time() - started) / 60:.0f} min", flush=True)
    if arguments.evaluate:
        model.eval()
        run_predictions(arguments, processor, model, parts)


# ---------------------------------------------------------------- prediction


@torch.inference_mode()
def predict_image(model, conversation, image, user_text, device):
    """Probabilities per class of each dimension, asking the questions one after the other on one KV cache."""
    prefix = {k: v.to(device) for k, v in conversation.prefix(image, user_text).items()}
    length = prefix["input_ids"].shape[1]
    next_ids = conversation.question_ids[common.DIMENSIONS[0]]
    output = model(**{**prefix, "input_ids": torch.cat([prefix["input_ids"], torch.tensor([next_ids], device=device)], 1),
                      "attention_mask": torch.ones(1, length + len(next_ids), dtype=torch.long, device=device)},
                   use_cache=True, logits_to_keep=1)
    length += len(next_ids)
    probabilities = {}
    for index, d in enumerate(common.DIMENSIONS):
        option_logits = output.logits[0, -1, conversation.letter_ids[d]].float()
        p = torch.softmax(option_logits, 0)
        probabilities[d] = {name: float(v) for name, v in zip(common.CLASSES[d], p)}
        if index == len(common.DIMENSIONS) - 1:
            break
        chosen = conversation.letter_ids[d][int(p.argmax())]
        next_ids = [chosen] + conversation.between_ids + conversation.question_ids[common.DIMENSIONS[index + 1]]
        ids = torch.tensor([next_ids], device=device)
        output = model(input_ids=ids, past_key_values=output.past_key_values,
                       attention_mask=torch.ones(1, length + len(next_ids), dtype=torch.long, device=device),
                       cache_position=torch.arange(length, length + len(next_ids), device=device),
                       use_cache=True, logits_to_keep=1)
        length += len(next_ids)
    return probabilities


def run_predictions(arguments, processor, model, parts):
    metric = common.load_metric(arguments.data_dir)
    conversation = Conversation(processor)
    device = next(model.parameters()).device
    predictions_dir = arguments.output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for part in arguments.parts:
        rows = parts[part] if arguments.predict_limit == 0 else parts[part].head(arguments.predict_limit)
        variants = [("context", True), ("no_context", False)] if part == "holdout" else [("context", True)]
        for name, with_context in variants:
            answers, started = {}, time.time()
            path = predictions_dir / f"{part}_{name}.jsonl"
            with path.open("w") as output:
                for row in tqdm(rows.to_dict("records"), desc=f"{part} {name}", mininterval=60):
                    image = common.prepare_image(row["image_file"], arguments.image_size)
                    user_text = common.user_prompt(row, True) if with_context else ""
                    probabilities = predict_image(model, conversation, image, user_text, device)
                    answers[row["image_id"]] = {d: max(p, key=p.get) for d, p in probabilities.items()}
                    output.write(json.dumps({"image_id": row["image_id"], "probabilities": probabilities}) + "\n")
            speed = len(rows) / (time.time() - started)
            print(f"\n===== {part} {name}: {len(rows)} images, {speed:.2f} images/s", flush=True)
            if part == "holdout":
                common.print_metrics(common.score_rows(metric, rows, answers, {}))
            if part == "dev" and with_context:
                write_submission(arguments, answers)


def write_submission(arguments, answers):
    sample = pd.read_csv(arguments.data_dir / "sample_submission.csv", dtype=str, keep_default_na=False)
    submission = sample.set_index("image_id")
    predicted = [image_id for image_id in submission.index if image_id in answers]
    submission.loc[predicted, list(common.DIMENSIONS)] = [[answers[i][d] for d in common.DIMENSIONS] for i in predicted]
    for d in common.DIMENSIONS:
        assert submission[d].isin(common.CLASSES[d]).all(), f"{d} has an invalid cell"
    path = arguments.output_dir / "submission_lora_atomic.csv"
    submission.reset_index().to_csv(path, index=False)
    print(f"{path}: {len(predicted)} dev rows predicted, final rows keep the constants", flush=True)


def predict(arguments):
    """Predict with a trained adapter, or zero-shot with the plain model when no adapter is given."""
    from zero_shot import load_model

    processor, model = load_model(arguments)
    if arguments.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(arguments.adapter))
    model.eval()
    print(f"between answer and next question: {Conversation(processor).between!r}", flush=True)
    run_predictions(arguments, processor, model, load_parts(arguments.data_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/lora_atomic"))
    parser.add_argument("--adapter", type=Path, help="for predict: the saved adapter folder")
    parser.add_argument("--model", default="swiss-ai/Apertus-v1.5-8B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="none")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--accumulation", type=int, default=16, help="images per optimizer step")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--no-context-share", type=float, default=1 / 3)
    parser.add_argument("--no-mirror", dest="mirror", action="store_false")
    parser.add_argument("--limit", type=int, default=0, help="train images, 0 for all (for quick tests)")
    parser.add_argument("--predict-limit", type=int, default=0, help="images per predicted part, 0 for all")
    parser.add_argument("--parts", nargs="+", default=["holdout", "dev"])
    parser.add_argument("--no-evaluate", dest="evaluate", action="store_false")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=common.SEED)
    arguments = parser.parse_args()
    train(arguments) if arguments.command == "train" else predict(arguments)


if __name__ == "__main__":
    main()
