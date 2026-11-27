"""Zero-shot test: the untrained Apertus v1.5 on a sample of the holdout, scored with metric.py.

Measures what we need to plan: the score without training, vision tokens per image, GPU memory, images per second.

    python src/zero_shot.py --data-dir ~/galaxus/data --limit 100
    python src/zero_shot.py --data-dir ~/galaxus/data --limit 100 --guideline   # with labeling_guideline.md
    python src/zero_shot.py --data-dir ~/galaxus/data --limit 100 --no-context  # without the product context
"""

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

import common


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="folder with train.csv, metric.py, images")
    parser.add_argument("--model", default="swiss-ai/Apertus-v1.5-8B")
    parser.add_argument("--revision", default=None, help="model revision (commit hash) for reproducibility")
    parser.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--image-size", type=int, default=512, help="images scaled to about this squared")
    parser.add_argument("--limit", type=int, default=100, help="holdout images to predict, 0 for all")
    parser.add_argument("--guideline", action="store_true", help="put labeling_guideline.md into the prompt")
    parser.add_argument("--no-context", action="store_true", help="leave out the product context")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--output", type=Path, default=Path("outputs/zero_shot.jsonl"))
    return parser.parse_args()


def load_model(arguments):
    """Apertus with its processor, quantized so it fits on a 12 GB card."""
    quantization = None
    if arguments.quantization == "4bit":
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
    elif arguments.quantization == "8bit":
        quantization = BitsAndBytesConfig(load_in_8bit=True)
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


def build_messages(row, image, prompt, with_context):
    return [
        {"role": "system", "content": [{"type": "text", "text": prompt}]},
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": common.user_prompt(row, with_context)}],
        },
    ]


def main():
    arguments = parse_arguments()
    metric = common.load_metric(arguments.data_dir)
    labeled = common.read_part(arguments.data_dir / "train.csv")
    _, holdout = common.split_holdout(labeled)
    rows = holdout if arguments.limit == 0 else holdout.head(arguments.limit)

    guideline = (arguments.data_dir / "labeling_guideline.md").read_text() if arguments.guideline else None
    prompt = common.system_prompt(guideline)
    with_context = not arguments.no_context

    processor, model = load_model(arguments)
    print(f"model loaded, {torch.cuda.memory_allocated() / 1024**3:.1f} GB on the GPU")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    answers, prompt_tokens, generated_tokens = {}, [], []
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    with arguments.output.open("w") as output:
        for row in tqdm(rows.to_dict("records"), desc="zero-shot"):
            image = common.prepare_image(row["image_file"], arguments.image_size)
            inputs = processor.apply_chat_template(
                build_messages(row, image, prompt, with_context),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=arguments.max_new_tokens, do_sample=False)
            new_tokens = generated[0, inputs["input_ids"].shape[1] :]
            text = processor.decode(new_tokens, skip_special_tokens=True)
            answers[row["image_id"]] = common.parse_answer(text)
            prompt_tokens.append(inputs["input_ids"].shape[1])
            generated_tokens.append(len(new_tokens))
            output.write(json.dumps({"image_id": row["image_id"], "answer": text, "size": image.size}) + "\n")
    elapsed = time.time() - started

    # A cell without a valid answer gets the most frequent human label, so the score stays comparable.
    fallback = {d: labeled.loc[labeled[f"{d}_source"] == "human", d].mode()[0] for d in common.DIMENSIONS}
    invalid = {d: sum(answer[d] is None for answer in answers.values()) for d in common.DIMENSIONS}

    print(f"\n{len(rows)} holdout images | context {'on' if with_context else 'off'} | "
          f"guideline {'on' if guideline else 'off'} | {arguments.quantization} | image size {arguments.image_size}")
    print(f"speed {len(rows) / elapsed:.2f} images/s (batch 1, transformers) | "
          f"peak GPU memory {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
    print(f"prompt tokens median {sorted(prompt_tokens)[len(prompt_tokens) // 2]} (system prompt + image + context) | "
          f"generated tokens median {sorted(generated_tokens)[len(generated_tokens) // 2]}")
    print(f"invalid answers per dimension: {invalid}\n")
    common.print_metrics(common.score_rows(metric, rows, answers, fallback))
    print(f"\nraw answers in {arguments.output}")


if __name__ == "__main__":
    main()
