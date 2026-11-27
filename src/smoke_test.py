"""Smoke test without competition data: does Apertus load, how much memory, how many vision tokens, how fast.

Uses a synthetic product image, so it runs before the Kaggle download.

    python src/smoke_test.py --image-sizes 256 384 512
"""

import argparse
import time

import torch
from PIL import Image, ImageDraw

import common
from zero_shot import ANSWER_PREFIX, count_image_tokens, encode, load_model


def synthetic_image():
    """A grey box with a blue label and some text on a white background."""
    image = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((250, 150, 550, 500), fill=(120, 120, 120))
    draw.rectangle((300, 250, 500, 330), fill=(40, 80, 200))
    draw.text((320, 280), "BIO 100% Natur", fill="white")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="swiss-ai/Apertus-v1.5-8B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--image-sizes", type=int, nargs="+", default=[256, 384, 512])
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()

    started = time.time()
    processor, model = load_model(arguments)
    print(f"loaded in {time.time() - started:.0f} s, {torch.cuda.memory_allocated() / 1024**3:.1f} GB on the GPU")

    prompt = common.system_prompt()
    row = {"product_title": "Bio Haferflocken 500 g", "brand": "Demo", "product_type": "Cereals",
           "category": "Food", "subcategory": "Breakfast"}
    for image_size in arguments.image_sizes:
        image = common.resize_to_area(synthetic_image(), image_size)
        messages = [
            {"role": "system", "content": prompt},  # the Apertus template wants a plain string
            {"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": common.user_prompt(row)}]},
        ]
        inputs = encode(processor, messages, image, model.device)
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for _ in range(arguments.repeats):
            torch.cuda.synchronize()
            start = time.time()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=120, do_sample=False)
            torch.cuda.synchronize()
            timings.append(time.time() - start)
        new_tokens = generated[0, inputs["input_ids"].shape[1]:]
        text = ANSWER_PREFIX + processor.decode(new_tokens, skip_special_tokens=True)
        print(f"generated {len(new_tokens)} tokens, parsed: {common.parse_answer(text)}")
        print(f"\nimage {image.size[0]}x{image.size[1]} | tokens {inputs['input_ids'].shape[1]}, "
              f"image {count_image_tokens(inputs['input_ids'], model)} | "
              f"{min(timings):.2f} s per image | peak {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
        print(f"keys in inputs: {sorted(inputs.keys())}")
        print(f"answer: {text[:400]}")


if __name__ == "__main__":
    main()
