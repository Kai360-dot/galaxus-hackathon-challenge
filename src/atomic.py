"""Atomic questions: one simple multiple-choice question per dimension, answered by the letter probabilities.

Instead of one JSON answer for all six dimensions, Apertus gets six short questions about the same image. Each option
carries a one-line explanation. The answer is not generated: we read the probability of each option letter at the
first answer position (as Vev does), so every answer is a valid class and comes with a probability per class.

The image and the product context are encoded once; the six questions reuse that KV cache.

    python src/atomic.py --data-dir ~/galaxus/data --limit 100 --variants context no_context
"""

import argparse
import copy
import json
import string
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

import common
from zero_shot import load_model

# dimension: (question, [(class, explanation), ...]). Paraphrased from the labeling guideline.
QUESTIONS = {
    "subject": (
        "What does the image show? Think of what the customer buys.",
        [
            ("full_product", "the sold item, whole, nothing cut off (a food pack counts as the product)"),
            ("detail_closeup", "a zoomed part, texture or swatch, or the product cut off at the frame"),
            ("accessories", "only an accessory or peripheral, not the main product"),
            ("packaging", "only the outer box, carton or bag, the product itself not visible"),
            ("package_and_product", "the product and its packaging both visible"),
            ("multipack_trade_unit", "several identical units in the same orientation"),
            ("delivery_scope", "the main product together with what comes with it (parts, cables, accessories)"),
            ("product_grouping", "several variants of the product (colors or sizes)"),
            ("multiview", "one product shown two or more times from different angles or states"),
            ("energy_label", "the EU energy efficiency label (A to G)"),
            ("infographic", "no product photo: text page, table, chart, diagram, icons, line drawing"),
            ("logo", "a single brand or company logo, nothing else"),
            ("placeholder", "a generic 'image coming soon' icon"),
        ],
    ),
    "background": (
        "What is behind the product?",
        [
            ("white", "pure white, a clean cutout, or no backdrop visible at all"),
            ("color", "a plain uniform backdrop that is not white: grey, cream, black, any color or gradient"),
            ("scene_lifestyle", "a real environment: a room, outdoors, a table or surface, props, a person"),
        ],
    ),
    "perspective": (
        "Find the front of the product. Seen from above with the camera at 6 o'clock, where does the front point?",
        [
            ("front_head_on", "straight at the camera, at product height"),
            ("front_angle_left", "toward the bottom left (7 to 8 o'clock)"),
            ("front_angle_right", "toward the bottom right (4 to 5 o'clock)"),
            ("back_head_on", "away from the camera, we see the back straight on"),
            ("back_angle_left", "toward the top right, we see the back (1 to 2 o'clock)"),
            ("back_angle_right", "toward the top left, we see the back (10 to 11 o'clock)"),
            ("left_side", "straight left, a pure side profile (9 o'clock)"),
            ("right_side", "straight right, a pure side profile (3 o'clock)"),
            ("top_down", "the camera looks straight down onto the top"),
            ("bottom_up", "the camera looks straight up at the underside"),
        ],
    ),
    "individuals": (
        "Is a real person or animal in the photo? Printed, drawn or on a screen does not count.",
        [
            ("individual_present", "yes, a real person, body part or animal is in the scene"),
            ("none", "no"),
        ],
    ),
    "text_classification": (
        "Which kind of text or graphic element was added digitally on top of the photo? "
        "Text printed on the product or pack does not count, except a nutrition table.",
        [
            ("nutritional_info", "a nutrition table (energy, fat, protein per 100 g, or similar)"),
            ("technical_specs", "dimension lines, measurements, size tags, callouts to parts"),
            ("promotional_text", "marketing claims, slogans, sale banners, 'NEW' badges"),
            ("certification_seal", "trust marks, certificates, eco-labels, award badges"),
            ("other", "other added elements: icons, pictograms, stickers, watermarks"),
            ("none", "nothing added on top"),
        ],
    ),
    "text_language": (
        "Which language dominates the visible words that say something? "
        "Brand and model names, technical terms and numbers do not count.",
        [
            ("German", "German"),
            ("English", "English"),
            ("French", "French"),
            ("Italian", "Italian"),
            ("Dutch", "Dutch"),
            ("other", "another language"),
            ("none", "no such words: only brands, technical terms, numbers or no text"),
        ],
    ),
}

SYSTEM_PROMPT = (
    "You answer questions about one product image of an online shop. "
    "Reply with the letter of the best option only."
)
QUESTION_MARKER = "<<QUESTION>>"
ANSWER_PREFIX = "Answer:"
VARIANTS = {"context": True, "no_context": False}


def question_text(dimension):
    question, options = QUESTIONS[dimension]
    lines = [question] + [f"{letter}) {explanation}" for letter, (_, explanation) in zip(string.ascii_uppercase, options)]
    return "\n".join(lines)


def letter_token_ids(tokenizer, count):
    """Per option letter the token ids it may start with: 'A' and ' A'."""
    ids = []
    for letter in string.ascii_uppercase[:count]:
        candidates = set()
        for text in (letter, " " + letter):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) == 1:
                candidates.add(encoded[0])
        assert candidates, f"no single token for option {letter}"
        ids.append(sorted(candidates))
    return ids


class AtomicClassifier:
    def __init__(self, processor, model):
        self.processor, self.model = processor, model
        tokenizer = processor.tokenizer
        self.letters = {d: letter_token_ids(tokenizer, len(options)) for d, (_, options) in QUESTIONS.items()}
        self.suffix_ids = {}

    def split_prompt(self, user_text):
        """Chat-templated text before the question (system, image, context) and after it (question, answer opening)."""
        content = [{"type": "image"}, {"type": "text", "text": user_text + QUESTION_MARKER}]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                  enable_thinking=False)
        prefix, suffix = text.split(QUESTION_MARKER)
        return prefix, suffix + ANSWER_PREFIX

    def option_logits(self, logits, dimension):
        """One logit per option: logsumexp over the tokens its letter may start with."""
        return torch.stack([torch.logsumexp(logits[ids], dim=0) for ids in self.letters[dimension]])

    @torch.inference_mode()
    def classify(self, image, user_text):
        """Per dimension the probability of each class, with the image and context encoded once."""
        prefix, suffix_template = self.split_prompt(user_text)
        inputs = self.processor(text=[prefix], images=[image], return_tensors="pt").to(self.model.device)
        prefix_length = inputs["input_ids"].shape[1]
        cache = self.model(**inputs, use_cache=True, logits_to_keep=1).past_key_values
        probabilities = {}
        for dimension, (_, options) in QUESTIONS.items():
            suffix = question_text(dimension) + suffix_template
            suffix_ids = self.processor.tokenizer(suffix, add_special_tokens=False, return_tensors="pt")["input_ids"]
            suffix_ids = suffix_ids.to(self.model.device)
            length = suffix_ids.shape[1]
            output = self.model(
                input_ids=suffix_ids,
                past_key_values=copy.deepcopy(cache),
                attention_mask=torch.ones(1, prefix_length + length, dtype=torch.long, device=self.model.device),
                cache_position=torch.arange(prefix_length, prefix_length + length, device=self.model.device),
                use_cache=True,
                logits_to_keep=1,
            )
            option_probabilities = torch.softmax(self.option_logits(output.logits[0, -1].float(), dimension), dim=0)
            probabilities[dimension] = {name: p for (name, _), p in zip(options, option_probabilities.tolist())}
        return probabilities, prefix_length

    @torch.inference_mode()
    def classify_uncached(self, image, user_text, dimension):
        """The same question without the cache, to check that the cached path gives the same answer."""
        prefix, suffix_template = self.split_prompt(user_text)
        text = prefix + question_text(dimension) + suffix_template
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        logits = self.model(**inputs, logits_to_keep=1).logits[0, -1].float()
        options = QUESTIONS[dimension][1]
        probabilities = torch.softmax(self.option_logits(logits, dimension), dim=0)
        return {name: p for (name, _), p in zip(options, probabilities.tolist())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", default="swiss-ai/Apertus-v1.5-8B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=["no_context"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    arguments = parser.parse_args()

    metric = common.load_metric(arguments.data_dir)
    labeled = common.read_part(arguments.data_dir / "train.csv")
    _, holdout = common.split_holdout(labeled)
    rows = holdout if arguments.limit == 0 else holdout.head(arguments.limit)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    processor, model = load_model(arguments)
    classifier = AtomicClassifier(processor, model)
    print("example question:\n" + question_text("individuals") + "\n")

    # The cached path must give the same answer as one plain forward pass.
    first = rows.iloc[0]
    image = common.prepare_image(first["image_file"], arguments.image_size)
    cached, _ = classifier.classify(image, common.user_prompt(first, True))
    for dimension in ("subject", "perspective"):
        uncached = classifier.classify_uncached(image, common.user_prompt(first, True), dimension)
        gap = max(abs(cached[dimension][name] - uncached[name]) for name in uncached)
        print(f"cache check {dimension}: largest probability gap {gap:.4f}")

    for name in arguments.variants:
        with_context = VARIANTS[name]
        answers, lengths = {}, []
        output_path = arguments.output_dir / f"atomic_{name}.jsonl"
        started = time.time()
        with output_path.open("w") as output:
            for row in tqdm(rows.to_dict("records"), desc=name, mininterval=30):
                image = common.prepare_image(row["image_file"], arguments.image_size)
                user_text = common.user_prompt(row, with_context) + "\n\n" if with_context else ""
                probabilities, prefix_length = classifier.classify(image, user_text)
                answers[row["image_id"]] = {d: max(p, key=p.get) for d, p in probabilities.items()}
                lengths.append(prefix_length)
                output.write(json.dumps({"image_id": row["image_id"], "probabilities": probabilities}) + "\n")
        elapsed = time.time() - started
        print(f"\n===== atomic {name}: {len(rows)} holdout images | {arguments.quantization} | "
              f"image size {arguments.image_size} | prefix tokens median {sorted(lengths)[len(lengths) // 2]}")
        print(f"speed {len(rows) / elapsed:.2f} images/s (batch 1, transformers, 6 questions on one cache)")
        fallback = {d: QUESTIONS[d][1][0][0] for d in common.DIMENSIONS}
        common.print_metrics(common.score_rows(metric, rows, answers, fallback))
        print(f"probabilities in {output_path}", flush=True)


if __name__ == "__main__":
    main()
