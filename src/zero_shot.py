"""Zero-shot test: the untrained Apertus v1.5 on a sample of the holdout, scored with metric.py.

Measures what we need to plan: the score without training, vision tokens per image, GPU memory, images per second.

    python src/zero_shot.py --data-dir ~/galaxus/data --limit 100 --variants plain guideline no_context

Variants share one loaded model:
    plain       short prompt with the class lists, product context on
    guideline   plain + labeling_guideline.md in the system prompt
    no_context  plain without the product context (production may lack it)
"""

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

import common


# name: (labeling guideline in the prompt, product context in the request)
VARIANTS = {"plain": (False, True), "guideline": (True, True), "no_context": (False, False)}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="folder with train.csv, metric.py, images")
    parser.add_argument("--model", default="swiss-ai/Apertus-v1.5-8B")
    parser.add_argument("--revision", default=None, help="model revision (commit hash) for reproducibility")
    parser.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--image-size", type=int, default=512, help="images scaled to about this squared")
    parser.add_argument("--limit", type=int, default=100, help="holdout images to predict, 0 for all")
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=["plain"])
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def load_model(arguments):
    """Apertus with its processor, quantized so it fits on a 12 GB card."""
    # The vision and audio tokenizers are precision-sensitive and stay in float32 (model card), so they are not
    # quantized; neither is the LM head.
    keep_full_precision = ["vision_tokenizer", "audio_tokenizer", "lm_head"]
    quantization = None
    if arguments.quantization == "4bit":
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=keep_full_precision,
        )
    elif arguments.quantization == "8bit":
        quantization = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=keep_full_precision)
    processor = AutoProcessor.from_pretrained(arguments.model, revision=arguments.revision)
    model = AutoModelForMultimodalLM.from_pretrained(
        arguments.model,
        revision=arguments.revision,
        dtype=torch.bfloat16,
        quantization_config=quantization,
        device_map="cuda:0",
    )
    model.eval()
    return processor, model


def count_image_tokens(input_ids, model):
    """The vision tokens of a request: ids from the image token offset on (config.json: image_token_offset)."""
    offset = model.config.image_token_offset
    audio_offset = getattr(model.config, "audio_token_offset", None)
    is_image = input_ids >= offset
    if audio_offset is not None:
        is_image &= input_ids < audio_offset
    return is_image.sum().item() or (input_ids == model.config.image_token_id).sum().item()


# Apertus starts deliberating (<|inner_prefix|> ...) even with enable_thinking=False. Opening the answer ourselves
# leaves it no room for that: it can only continue the JSON.
ANSWER_PREFIX = '{"subject": "'


def encode(processor, messages, image, device):
    """Tokens and pixels of a request, with the assistant answer already opened by ANSWER_PREFIX."""
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return processor(text=[text + ANSWER_PREFIX], images=[image], return_tensors="pt").to(device)


def build_messages(row, image, prompt, with_context):
    return [
        {"role": "system", "content": prompt},  # the Apertus template wants a plain string
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": common.user_prompt(row, with_context)}],
        },
    ]


def run_variant(name, arguments, processor, model, rows, labeled, metric):
    """Predict the rows with one prompt variant, then print speed, tokens and scores."""
    with_guideline, with_context = VARIANTS[name]
    guideline = (arguments.data_dir / "labeling_guideline.md").read_text() if with_guideline else None
    prompt = common.system_prompt(guideline)
    output_path = arguments.output_dir / f"zero_shot_{name}.jsonl"

    answers, prompt_tokens, generated_tokens, image_tokens = {}, [], [], []
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    with output_path.open("w") as output:
        for row in tqdm(rows.to_dict("records"), desc=name, mininterval=30):
            image = common.prepare_image(row["image_file"], arguments.image_size)
            inputs = encode(processor, build_messages(row, image, prompt, with_context), image, model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=arguments.max_new_tokens, do_sample=False)
            new_tokens = generated[0, inputs["input_ids"].shape[1] :]
            text = ANSWER_PREFIX + processor.decode(new_tokens, skip_special_tokens=True)
            answers[row["image_id"]] = common.parse_answer(text)
            prompt_tokens.append(inputs["input_ids"].shape[1])
            image_tokens.append(int(count_image_tokens(inputs["input_ids"], model)))
            generated_tokens.append(len(new_tokens))
            output.write(json.dumps({"image_id": row["image_id"], "answer": text, "size": image.size}) + "\n")
    elapsed = time.time() - started

    # A cell without a valid answer gets the most frequent human label, so the score stays comparable.
    fallback = {d: labeled.loc[labeled[f"{d}_source"] == "human", d].mode()[0] for d in common.DIMENSIONS}
    invalid = {d: sum(answer[d] is None for answer in answers.values()) for d in common.DIMENSIONS}
    median = lambda values: sorted(values)[len(values) // 2]

    print(f"\n===== {name}: {len(rows)} holdout images | context {'on' if with_context else 'off'} | "
          f"guideline {'on' if guideline else 'off'} | {arguments.quantization} | image size {arguments.image_size}")
    print(f"speed {len(rows) / elapsed:.2f} images/s (batch 1, transformers) | "
          f"peak GPU memory {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
    print(f"tokens per request median {median(prompt_tokens)}, of which image {median(image_tokens)} | "
          f"generated tokens median {median(generated_tokens)}")
    print(f"invalid answers per dimension: {invalid}\n")
    common.print_metrics(common.score_rows(metric, rows, answers, fallback))
    print(f"raw answers in {output_path}", flush=True)


def main():
    arguments = parse_arguments()
    metric = common.load_metric(arguments.data_dir)
    labeled = common.read_part(arguments.data_dir / "train.csv")
    _, holdout = common.split_holdout(labeled)
    rows = holdout if arguments.limit == 0 else holdout.head(arguments.limit)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    processor, model = load_model(arguments)
    print(f"model loaded in {time.time() - started:.0f} s, {torch.cuda.memory_allocated() / 1024**3:.1f} GB on the GPU")
    for name in arguments.variants:
        run_variant(name, arguments, processor, model, rows, labeled, metric)


if __name__ == "__main__":
    main()
