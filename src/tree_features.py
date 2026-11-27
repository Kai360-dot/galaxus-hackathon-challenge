"""The question tree's questions as features: P(yes) of each yes/no question, read from the next-token logits.

The question tree on main asks Apertus 22 small yes/no questions per image (src/question_tree/questions.py) and turns
the answers into labels with hand-written rules. Here the same questions feed the heads instead: Apertus sees the
image once, every question is appended to that KV cache on its own (the cache is cut back after each), and the
probability of "yes" against "no" at the answer position becomes one feature. The six atomic multiple-choice
questions of atomic.py add their option probabilities. 22 + 41 = 63 numbers per image.

    python src/tree_features.py --data-dir ~/galaxus/data --features-dir results/tree_4bit --parts holdout \
        --quantization 4bit
    python src/tree_features.py ... --parts train --sample 1500
"""

import argparse
import copy
import importlib.util
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

import common
from atomic import QUESTIONS as ATOMIC, question_text
from heads import load_parts

TREE_QUESTIONS_FILE = Path(__file__).parent / "question_tree" / "questions.py"
ANSWER_PREFIX = "Answer:"


def tree_questions():
    """questions.py of the question tree, loaded as a file: the package needs API settings at import."""
    spec = importlib.util.spec_from_file_location("tree_questions", TREE_QUESTIONS_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def token_ids(tokenizer, words):
    """Single-token ids of the given words with and without a leading space."""
    ids = set()
    for word in words:
        for text in (word, " " + word):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) == 1:
                ids.add(encoded[0])
    assert ids, f"no single token for {words}"
    return sorted(ids)


class QuestionFeatures:
    def __init__(self, processor, model):
        self.processor, self.model = processor, model
        self.tree = tree_questions()
        tokenizer = processor.tokenizer
        self.yes = token_ids(tokenizer, ["yes", "Yes"])
        self.no = token_ids(tokenizer, ["no", "No"])
        self.letters = {d: [token_ids(tokenizer, [chr(65 + i)]) for i in range(len(options))]
                        for d, (_, options) in ATOMIC.items()}
        marker = "<<Q>>"
        messages = [{"role": "system", "content": self.tree.SYSTEM},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": marker}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        self.prefix_text, after = text.split(marker)
        self.suffixes = []  # (kind, key, token ids)
        for key in self.tree.TREE_YES_NO:
            question = self.tree.QUESTIONS[key][0] + " Answer yes or no."
            self.suffixes.append(("yes_no", key, tokenizer(question + after + ANSWER_PREFIX, add_special_tokens=False)["input_ids"]))
        for d in common.DIMENSIONS:
            self.suffixes.append(("choice", d, tokenizer(question_text(d) + after + ANSWER_PREFIX, add_special_tokens=False)["input_ids"]))
        self.names = [key for kind, key, _ in self.suffixes if kind == "yes_no"] + [
            f"{d}={name}" for d in common.DIMENSIONS for name, _ in ATOMIC[d][1]]

    @torch.inference_mode()
    def features(self, image):
        """63 numbers: P(yes) per yes/no question, then the option probabilities of the six atomic questions."""
        device = self.model.device
        inputs = self.processor(text=[self.prefix_text], images=[image], return_tensors="pt").to(device)
        length = inputs["input_ids"].shape[1]
        cache = self.model(**inputs, use_cache=True, logits_to_keep=1).past_key_values
        crop = hasattr(cache, "crop")
        yes_no, choices = [], []
        for kind, key, ids in self.suffixes:
            branch = cache if crop else copy.deepcopy(cache)
            ids_tensor = torch.tensor([ids], device=device)
            output = self.model(input_ids=ids_tensor, past_key_values=branch,
                                attention_mask=torch.ones(1, length + len(ids), dtype=torch.long, device=device),
                                cache_position=torch.arange(length, length + len(ids), device=device),
                                use_cache=True, logits_to_keep=1)
            logits = output.logits[0, -1].float()
            if crop:
                cache.crop(length)  # back to image and system prompt for the next question
            if kind == "yes_no":
                pair = torch.stack([torch.logsumexp(logits[self.yes], 0), torch.logsumexp(logits[self.no], 0)])
                yes_no.append(float(torch.softmax(pair, 0)[0]))
            else:
                option_logits = torch.stack([torch.logsumexp(logits[i], 0) for i in self.letters[key]])
                choices += torch.softmax(option_logits, 0).tolist()
        return np.array(yes_no + choices, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, default=Path("results/tree_4bit"))
    parser.add_argument("--parts", nargs="+", default=["holdout"])
    parser.add_argument("--sample", type=int, default=0, help="random images per part, 0 for all")
    parser.add_argument("--model", default="swiss-ai/Apertus-v1.5-8B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--adapter", type=Path, help="a LoRA adapter folder (lora_atomic.py) to load on Apertus")
    arguments = parser.parse_args()

    from zero_shot import load_model

    processor, model = load_model(arguments)
    if arguments.adapter:  # our LoRA-finetuned Apertus answers the questions instead of the plain model
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(arguments.adapter)).eval()
    extractor = QuestionFeatures(processor, model)
    print(f"{len(extractor.suffixes)} questions per image, {len(extractor.names)} features", flush=True)
    parts = load_parts(arguments.data_dir)
    for part in arguments.parts:
        rows = parts[part]
        if arguments.sample:
            rows = rows.iloc[sorted(random.Random(common.SEED).sample(range(len(rows)), arguments.sample))]
        records = rows.to_dict("records")
        features, started = [], time.time()
        for row in tqdm(records, desc=part, mininterval=60):
            features.append(extractor.features(common.prepare_image(row["image_file"], arguments.image_size)))
        part_dir = arguments.features_dir / part
        part_dir.mkdir(parents=True, exist_ok=True)
        np.savez(part_dir / "shard_0000.npz", image_ids=np.array([r["image_id"] for r in records]),
                 mirrored=np.zeros(len(records), dtype=bool), context=np.ones(len(records), dtype=bool),
                 features=np.stack(features)[:, None, None], layers=np.array([0]), names=np.array(extractor.names))
        print(f"{part}: {len(records)} images, {len(records) / (time.time() - started):.2f} images/s", flush=True)


if __name__ == "__main__":
    main()
