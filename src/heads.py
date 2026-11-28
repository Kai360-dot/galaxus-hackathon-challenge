"""Classification heads on a frozen Apertus: one forward pass per image, six linear heads on its hidden states.

Two steps:

    # 1. Hidden states of every image, once (the expensive part, resumable).
    python src/heads.py extract --data-dir ~/galaxus/data --features-dir features --parts holdout train dev

    # 2. Heads on those features (seconds per try), scored on the holdout; writes a dev submission.
    python src/heads.py fit --data-dir ~/galaxus/data --features-dir features

Per image we keep, for a few layers, the hidden state of the last prompt token and the mean over the vision tokens.
Train images with a left or right perspective and no text also get a mirrored copy (left and right swapped).

Context dropout: in production the product context can be missing, so a third of the train images (fixed seed) is
also extracted without it, with the same labels. The holdout is extracted both ways, to score both cases.
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

import common

SYSTEM_PROMPT = (
    "You classify one product image of an online shop along six dimensions: what it shows, the background, "
    "the perspective, whether a real person or animal is present, the kind of text overlay and the text language."
)


def specific_prompt():
    """The six atomic questions with their explained options, so the model knows what to look for."""
    from atomic import question_text

    questions = "\n\n".join(question_text(d) for d in common.DIMENSIONS)
    return ("You classify one product image of an online shop by answering these six questions about it:\n\n"
            + questions)


PROMPTS = {"vague": lambda: SYSTEM_PROMPT, "specific": specific_prompt}
DEFAULT_LAYERS = (8, 16, 24, 32)
SHARD_SIZE = 500
NO_CONTEXT_SHARE = 1 / 3

MIRRORED_PERSPECTIVES = {
    "front_angle_left": "front_angle_right", "front_angle_right": "front_angle_left",
    "back_angle_left": "back_angle_right", "back_angle_right": "back_angle_left",
    "left_side": "right_side", "right_side": "left_side",
}


# ---------------------------------------------------------------- parts and targets


def load_parts(data_dir):
    """train (without holdout), holdout and dev rows, and final rows once they are decrypted."""
    labeled = common.read_part(data_dir / "train.csv")
    train, holdout = common.split_holdout(labeled)
    dev = common.read_part(data_dir / "dev.csv")
    parts = {"train": train, "holdout": holdout, "dev": dev}
    final_csv = data_dir / "final" / "final.csv"  # after decrypting final_encrypted.bin into data/final
    if final_csv.exists():
        parts["final"] = common.read_part(final_csv)
    return parts


def is_mirrorable(row):
    """A human left or right perspective on an image without text: mirroring keeps every label valid."""
    return (
        row["perspective_source"] == "human"
        and row["perspective"] in MIRRORED_PERSPECTIVES
        and row["text_classification"] == "none"
        and row["text_language"] == "none"
    )


def extraction_jobs(part, rows, mirror, no_context_share):
    """(image_id, row, mirrored, with_context) per feature row to extract.

    train: every image with context, mirrored copies, and a share of the images again without context.
    holdout: every image with and without context. dev: with context.
    """
    records = rows.to_dict("records")
    jobs = [(row["image_id"], row, False, True) for row in records]
    if part == "train":
        if mirror:
            jobs += [(row["image_id"], row, True, True) for row in records if is_mirrorable(row)]
        dropped = random.Random(common.SEED).sample(records, round(no_context_share * len(records)))
        jobs += [(row["image_id"], row, False, False) for row in dropped]
    elif part == "holdout":
        jobs += [(row["image_id"], row, False, False) for row in records]
    return jobs


# ---------------------------------------------------------------- extraction


def vision_token_mask(input_ids, config):
    """True at the vision tokens: the image placeholder, or ids in the vision token range."""
    mask = input_ids == config.image_token_id
    offset, audio_offset = config.image_token_offset, getattr(config, "audio_token_offset", None)
    in_range = input_ids >= offset
    if audio_offset is not None:
        in_range &= input_ids < audio_offset
    return mask | in_range


def encode(processor, row, image, with_context, system_prompt=SYSTEM_PROMPT):
    user_text = common.user_prompt(row, with_context) + "\n\nClassify this image." if with_context else "Classify this image."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}]},
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return processor(text=[text], images=[image], return_tensors="pt")


@torch.inference_mode()
def image_features(model, processor, row, image, layers, with_context, system_prompt=SYSTEM_PROMPT):
    """(layers, 2, hidden): per layer the last-token state and the mean over vision tokens, float16."""
    inputs = encode(processor, row, image, with_context, system_prompt).to(model.device)
    output = model(**inputs, output_hidden_states=True, logits_to_keep=1)
    vision = vision_token_mask(inputs["input_ids"][0], model.config)
    assert vision.sum() > 16, f"only {int(vision.sum())} vision tokens found"
    states = []
    for layer in layers:
        hidden = output.hidden_states[layer][0]
        states.append(torch.stack([hidden[-1], hidden[vision].mean(dim=0)]))
    return torch.stack(states).float().cpu().numpy().astype(np.float16), int(vision.sum())


def extract(arguments):
    from zero_shot import load_model

    parts = load_parts(arguments.data_dir)
    processor, model = load_model(arguments)
    if arguments.truncate:  # skip the layers after the deepest one read; keep one more so its state is not the final norm
        language = model.model.language_model
        keep = max(arguments.layers) + 1
        language.layers = torch.nn.ModuleList(list(language.layers)[:keep])
        if hasattr(model.config, "text_config"):
            model.config.text_config.num_hidden_layers = keep
    system_prompt = PROMPTS[arguments.prompt]()
    print(f"model loaded, {torch.cuda.memory_allocated() / 1024**3:.1f} GB on the GPU | prompt {arguments.prompt}, "
          f"{len(processor.tokenizer(system_prompt)['input_ids'])} system tokens", flush=True)
    for part in arguments.parts:
        rows = parts[part] if arguments.limit == 0 else parts[part].head(arguments.limit)
        jobs = extraction_jobs(part, rows, arguments.mirror, arguments.no_context_share)
        part_dir = arguments.features_dir / part
        part_dir.mkdir(parents=True, exist_ok=True)
        started, done = time.time(), 0
        for shard_start in range(0, len(jobs), SHARD_SIZE):
            shard_path = part_dir / f"shard_{shard_start // SHARD_SIZE:04d}.npz"
            if shard_path.exists():
                continue  # resume: this shard is already extracted
            shard = jobs[shard_start : shard_start + SHARD_SIZE]
            features, vision_counts = [], []
            for image_id, row, mirrored, with_context in tqdm(
                shard, desc=f"{part} shard {shard_start // SHARD_SIZE}", mininterval=30
            ):
                image = common.prepare_image(row["image_file"], arguments.image_size, mirrored=mirrored)
                feature, vision_count = image_features(model, processor, row, image, arguments.layers, with_context,
                                                       system_prompt)
                features.append(feature)
                vision_counts.append(vision_count)
            np.savez(
                shard_path,
                image_ids=np.array([job[0] for job in shard]),
                mirrored=np.array([job[2] for job in shard]),
                context=np.array([job[3] for job in shard]),
                features=np.stack(features),
                layers=np.array(arguments.layers),
            )
            done += len(shard)
            rate = done / (time.time() - started)
            print(f"{part}: {shard_start + len(shard)} / {len(jobs)} | {rate:.2f} images/s | "
                  f"vision tokens median {int(np.median(vision_counts))}", flush=True)


# ---------------------------------------------------------------- heads


def load_features(features_dir, part):
    shards = sorted((features_dir / part).glob("shard_*.npz"))
    assert shards, f"no features for {part} in {features_dir}"
    loaded = [np.load(path) for path in shards]
    return (
        np.concatenate([shard["image_ids"] for shard in loaded]),
        np.concatenate([shard["mirrored"] for shard in loaded]),
        np.concatenate([shard["context"] for shard in loaded]),
        np.concatenate([shard["features"] for shard in loaded]),
        [int(layer) for layer in loaded[0]["layers"]],
    )


def targets(rows, image_ids, mirrored):
    """Per dimension the class index of each feature row, -1 where the label is a model prelabel."""
    by_id = rows.set_index("image_id")
    result = {}
    for dimension in common.DIMENSIONS:
        index = {name: i for i, name in enumerate(common.CLASSES[dimension])}
        values = []
        for image_id, flipped in zip(image_ids, mirrored):
            row = by_id.loc[image_id]
            label = row[dimension]
            if dimension == "perspective" and flipped:
                label = MIRRORED_PERSPECTIVES.get(label, label)  # front, back, top and bottom stay as they are
            values.append(index[label] if row[f"{dimension}_source"] == "human" else -1)
        result[dimension] = torch.tensor(values)
    return result


def select(features, layer_indices, pooling):
    """The chosen layers and poolings, flattened to one vector per image."""
    poolings = {"last": [0], "mean": [1], "both": [0, 1]}[pooling]
    return features[:, layer_indices][:, :, poolings].reshape(len(features), -1).astype(np.float32)


def make_head(inputs, hidden, classes):
    """A linear head, or one hidden layer when hidden > 0, with dropout."""
    if hidden:
        return torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(inputs, hidden), torch.nn.GELU(),
                                   torch.nn.Dropout(0.2), torch.nn.Linear(hidden, classes))
    return torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(inputs, classes))


def train_heads(x, y, hidden, epochs, device, seed):
    """One head per dimension (linear, or one hidden layer), class-weighted cross-entropy on human labels."""
    torch.manual_seed(seed)
    x = torch.tensor(x, device=device)
    mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True) + 1e-4
    x = (x - mean) / std
    heads = {}
    for dimension in common.DIMENSIONS:
        labels = y[dimension].to(device)
        keep = labels >= 0
        classes = len(common.CLASSES[dimension])
        counts = torch.bincount(labels[keep], minlength=classes).float()
        weights = (counts.clamp(min=1) ** -0.5) * (counts > 0)  # rare classes weigh more, absent ones not at all
        weights = weights / (weights * counts).sum() * counts.sum()
        head = make_head(x.shape[1], hidden, classes).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.05)
        loss_function = torch.nn.CrossEntropyLoss(weight=weights)
        xs, ys = x[keep], labels[keep]
        for _ in range(epochs):
            permutation = torch.randperm(len(xs), device=device)
            for start in range(0, len(xs), 512):
                batch = permutation[start : start + 512]
                optimizer.zero_grad()
                loss_function(head(xs[batch]), ys[batch]).backward()
                optimizer.step()
        heads[dimension] = head.eval()
    return heads, mean, std


@torch.inference_mode()
def predict(heads, mean, std, x, device):
    x = (torch.tensor(x, device=device) - mean) / std
    return {d: torch.softmax(head(x), dim=1).cpu().numpy() for d, head in heads.items()}


def answers_from(probabilities, image_ids):
    return {
        image_id: {d: common.CLASSES[d][int(probabilities[d][i].argmax())] for d in common.DIMENSIONS}
        for i, image_id in enumerate(image_ids)
    }


def fit(arguments):
    parts = load_parts(arguments.data_dir)
    metric = common.load_metric(arguments.data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ids, train_mirrored, train_context, train_features, layers = load_features(arguments.features_dir, "train")
    all_holdout_ids, _, holdout_context, all_holdout_features, _ = load_features(arguments.features_dir, "holdout")
    y = targets(parts["train"], train_ids, train_mirrored)
    # The main score is with context (as on Kaggle); the copies without context show what production loses.
    holdout_ids, holdout_features = all_holdout_ids[holdout_context], all_holdout_features[holdout_context]
    bare_ids, bare_features = all_holdout_ids[~holdout_context], all_holdout_features[~holdout_context]
    holdout_rows = parts["holdout"].set_index("image_id").loc[holdout_ids].reset_index()
    bare_rows = parts["holdout"].set_index("image_id").loc[bare_ids].reset_index()
    print(f"train {len(train_ids)} feature rows ({int(train_mirrored.sum())} mirrored, "
          f"{int((~train_context).sum())} without context) | holdout {len(holdout_ids)} with context, "
          f"{len(bare_ids)} without | layers {layers}")

    if arguments.only:
        only_layers, only_pooling, only_hidden = arguments.only.split(":")
        chosen = [layers.index(int(layer)) for layer in only_layers.split(",")]
        configurations, hiddens = [(chosen, only_pooling)], [int(only_hidden)]
    else:
        configurations = [([layers.index(l)], pooling) for l in layers for pooling in ("last", "mean", "both")]
        configurations.append((list(range(len(layers))), "both"))
        hiddens = arguments.hidden
    results = []
    for layer_indices, pooling in configurations:
        for hidden in hiddens:
            heads, mean, std = train_heads(select(train_features, layer_indices, pooling), y, hidden,
                                           arguments.epochs, device, arguments.seed)
            probabilities = predict(heads, mean, std, select(holdout_features, layer_indices, pooling), device)
            metrics = common.score_rows(metric, holdout_rows, answers_from(probabilities, holdout_ids), {})
            bare_score = float("nan")
            if len(bare_ids):
                bare_probabilities = predict(heads, mean, std, select(bare_features, layer_indices, pooling), device)
                bare_metrics = common.score_rows(metric, bare_rows, answers_from(bare_probabilities, bare_ids), {})
                bare_score = bare_metrics["overall"]["f1_macro"]
            name = f"layers {[layers[i] for i in layer_indices]} {pooling:4} {'mlp' if hidden else 'linear'}"
            results.append((metrics["overall"]["f1_macro"], name, layer_indices, pooling, hidden, metrics))
            print(f"{name:40} macro F1 {metrics['overall']['f1_macro']:.3f} (no context {bare_score:.3f}) | "
                  f"weighted {metrics['overall']['f1_weighted']:.3f}"
                  + "".join(f" | {d[:8]} {metrics[d]['f1_macro']:.2f}" for d in common.DIMENSIONS if d in metrics),
                  flush=True)

    best = max(results, key=lambda result: result[0])
    print(f"\nbest on the holdout: {best[1]}\n")
    common.print_metrics(best[5])

    # The best configuration, trained on train only (the holdout stays our honest check), saved for prediction.
    _, _, layer_indices, pooling, hidden, _ = best
    heads, mean, std = train_heads(select(train_features, layer_indices, pooling), y, hidden,
                                   arguments.epochs, device, arguments.seed)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    heads_path = arguments.output_dir / "heads_best.pt"
    torch.save({
        "layers": [layers[i] for i in layer_indices], "pooling": pooling, "hidden": hidden,
        "mean": mean.cpu(), "std": std.cpu(), "prompt": arguments.prompt, "image_size": arguments.image_size,
        "heads": {d: head.state_dict() for d, head in heads.items()},
    }, heads_path)
    print(f"heads saved to {heads_path}")
    if (arguments.features_dir / "dev").exists():
        dev_ids, _, _, dev_features, _ = load_features(arguments.features_dir, "dev")
        dev_answers = answers_from(predict(heads, mean, std, select(dev_features, layer_indices, pooling), device), dev_ids)
        sample = pd.read_csv(arguments.data_dir / "sample_submission.csv", dtype=str, keep_default_na=False)
        submission = sample.set_index("image_id")
        predicted = [image_id for image_id in submission.index if image_id in dev_answers]
        submission.loc[predicted, list(common.DIMENSIONS)] = [[dev_answers[i][d] for d in common.DIMENSIONS]
                                                              for i in predicted]
        for d in common.DIMENSIONS:
            assert submission[d].isin(common.CLASSES[d]).all(), f"{d} has an invalid cell"
        path = arguments.output_dir / "submission_heads.csv"
        submission.reset_index().to_csv(path, index=False)
        print(f"\n{path}: {len(predicted)} of {len(dev_ids)} dev rows predicted, final rows keep the constants")


def load_heads(path, device):
    saved = torch.load(path, map_location=device)
    inputs = saved["mean"].shape[1]
    heads = {}
    for d in common.DIMENSIONS:
        head = make_head(inputs, saved["hidden"], len(common.CLASSES[d]))
        head.load_state_dict(saved["heads"][d])
        heads[d] = head.to(device).eval()
    return saved, heads, saved["mean"].to(device), saved["std"].to(device)


def predict_command(arguments):
    """Classify images with Apertus and saved heads: given files, or a part of the data scored with metric.py."""
    from zero_shot import load_model

    processor, model = load_model(arguments)
    device = model.device
    saved, heads, mean, std = load_heads(arguments.heads, device)
    system_prompt = PROMPTS[saved["prompt"]]()
    layers, pooling = saved["layers"], saved["pooling"]
    image_size = saved.get("image_size", arguments.image_size)
    print(f"heads: layers {layers}, {pooling}, {'mlp ' + str(saved['hidden']) if saved['hidden'] else 'linear'}, "
          f"prompt {saved['prompt']} | {arguments.quantization} | image size {image_size}", flush=True)

    def classify(row, image_file, with_context):
        image = common.prepare_image(image_file, image_size)
        feature, _ = image_features(model, processor, row, image, layers, with_context, system_prompt)
        x = select(feature[None], list(range(len(layers))), pooling)
        return {d: p[0] for d, p in predict(heads, mean, std, x, device).items()}

    if arguments.images:
        empty = {"product_title": "", "brand": "", "product_type": "", "category": "", "subcategory": ""}
        for image_file in arguments.images:
            started = time.time()
            probabilities = classify(empty, image_file, False)
            print(f"\n{image_file} ({time.time() - started:.2f} s)")
            for d in common.DIMENSIONS:
                order = probabilities[d].argsort()[::-1][:2]
                print(f"  {d:20} " + ", ".join(f"{common.CLASSES[d][i]} {probabilities[d][i]:.2f}" for i in order))
        return

    parts = load_parts(arguments.data_dir)
    metric = common.load_metric(arguments.data_dir)
    for part in arguments.parts:
        rows = parts[part] if arguments.limit == 0 else parts[part].head(arguments.limit)
        for with_context in (True, False) if part == "holdout" else (True,):
            answers, started = {}, time.time()
            for row in tqdm(rows.to_dict("records"), desc=f"{part} context {with_context}", mininterval=30):
                probabilities = classify(row, row["image_file"], with_context)
                answers[row["image_id"]] = {d: common.CLASSES[d][int(p.argmax())] for d, p in probabilities.items()}
            speed = len(rows) / (time.time() - started)
            print(f"\n===== {part}, context {'on' if with_context else 'off'}: {len(rows)} images, "
                  f"{speed:.2f} images/s, peak {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB", flush=True)
            if part == "holdout":
                common.print_metrics(common.score_rows(metric, rows, answers, {}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["extract", "fit", "predict"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--features-dir", type=Path, default=Path("features"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", default="swiss-ai/Apertus-v1.5-8B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--parts", nargs="+", default=["holdout", "train", "dev"])
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="vague",
                        help="vague: one sentence naming the dimensions; specific: the six atomic questions")
    parser.add_argument("--truncate", action="store_true", help="extract: run Apertus only up to the deepest layer read")
    parser.add_argument("--limit", type=int, default=0, help="images per part, 0 for all (for quick tests)")
    parser.add_argument("--no-mirror", dest="mirror", action="store_false")
    parser.add_argument("--no-context-share", type=float, default=NO_CONTEXT_SHARE,
                        help="share of train images also extracted without the product context")
    parser.add_argument("--hidden", type=int, nargs="+", default=[0], help="0 for linear heads, else MLP width")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--only", help="fit one configuration, LAYERS:POOLING:HIDDEN, e.g. 24:both:1024")
    parser.add_argument("--heads", type=Path, default=Path("outputs/heads_best.pt"), help="for predict")
    parser.add_argument("--images", nargs="+", type=Path, help="for predict: image files, without context")
    parser.add_argument("--seed", type=int, default=common.SEED)
    arguments = parser.parse_args()
    {"extract": extract, "fit": fit, "predict": predict_command}[arguments.command](arguments)


if __name__ == "__main__":
    main()
