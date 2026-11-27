"""The strategies that were tested against the question tree and dropped.

Each function produces the stored file that the results notebook scores. They all call the Apertus endpoint,
so they are only needed to repeat an experiment. Stored answers are never asked again.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor

from .baseline_prompt import SYSTEM_PROMPT
from .client import image_message, post_chat, schema_format
from .config import CLASSES, DATA_DIR, DIMENSIONS, STORE
from .images import data_url, prepared_image
from .questions import QUESTIONS, SYSTEM, TREE_YES_NO, YES_NO

LABELS_FORMAT = schema_format({dimension: {"type": "string", "enum": CLASSES[dimension]} for dimension in DIMENSIONS})
CONTEXT = "Product: {product_title}\nBrand: {brand}\nProduct type: {product_type}\nCategory: {category} / {subcategory}"
DESCRIBE = (
    "Describe this product image factually in 6 to 9 short sentences. Cover: which objects are shown and how many; "
    "whether the main product is complete or cut off by the image border; whether packaging is visible; what is behind "
    "the product (pure white, a plain color, or a real scene); whether a real person or animal is visible; from which "
    "side the product is seen (front, back, left side, right side, from above, from below, at an angle); any text or "
    "graphics that were added on top of the photo; and the language of the visible words."
)
ROLE = (
    "Which is true? Answer product if the image shows the listed product itself, also when it is a set that is sold as "
    "one product. Answer part if the image shows only an accessory, spare part or component that belongs to the listed "
    "product. Answer set if the image shows the listed product together with additional separate items that come with it."
)


def _labels(text):
    """The six labels in a reply, None where the reply holds no valid class."""
    try:
        parsed = json.loads(text[text.find("{") : text.rfind("}") + 1])
    except ValueError:
        parsed = {}
    return {dimension: parsed.get(dimension) if parsed.get(dimension) in CLASSES[dimension] else None for dimension in DIMENSIONS}


def _image(row):
    return data_url(prepared_image(DATA_DIR / row["image_path"]))


def full_guideline(model, row):
    """One prompt with the complete labeling guideline as instructions."""
    guideline = (DATA_DIR / "labeling_guideline.md").read_text()
    system = "You label one product image of an online shop along six dimensions. Below is the complete labeling guideline that the human labelers followed. Apply it exactly. Answer with one JSON object that holds exactly one class per dimension.\n\n" + guideline
    body = {"model": model, "temperature": 0, "max_tokens": 120, "response_format": LABELS_FORMAT,
            "messages": [{"role": "system", "content": system}, image_message(_image(row), CONTEXT.format(**row))]}
    reply = post_chat(body)
    return {"labels": _labels(reply["choices"][0]["message"]["content"]), "prompt_tokens": reply["usage"]["prompt_tokens"]}


def thinking(model, row):
    """One prompt, answered by the thinking variant of the model (model name ending in -thinking)."""
    body = {"model": model, "temperature": 0, "max_tokens": 2500,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, image_message(_image(row), CONTEXT.format(**row) + "\n\nThink it through, then end your answer with the JSON object.")]}
    reply = post_chat(body)
    message = reply["choices"][0]["message"]
    return {"labels": _labels(message.get("content") or ""), "reasoning_chars": len(message.get("reasoning") or ""), "completion_tokens": reply["usage"]["completion_tokens"]}


def describe_then_classify(model, row):
    """Two requests: the model describes the image, then labels it from its own description without seeing it."""
    description = post_chat({"model": model, "temperature": 0, "max_tokens": 260, "messages": [image_message(_image(row), DESCRIBE)]})["choices"][0]["message"]["content"]
    first = "You label one product image of an online shop along six dimensions."
    system = SYSTEM_PROMPT.replace(first, first + " You do not see the image, you get a written description of it.")
    body = {"model": model, "temperature": 0, "max_tokens": 120, "response_format": LABELS_FORMAT,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": CONTEXT.format(**row) + "\n\nDescription of the image:\n" + description}]}
    return {"labels": _labels(post_chat(body)["choices"][0]["message"]["content"])}


def role_chain(model, row):
    """Name the object, say what it is used for, then decide its role against the listing. Tested to find accessories."""
    image_url = _image(row)

    def ask(text, allowed=None, max_tokens=16):
        body = {"model": model, "temperature": 0, "max_tokens": max_tokens, "messages": [{"role": "system", "content": SYSTEM}, image_message(image_url, text)]}
        if allowed:
            body["response_format"] = schema_format({"answer": {"type": "string", "enum": allowed}})
        content = post_chat(body)["choices"][0]["message"]["content"] or ""
        return json.loads(content)["answer"] if allowed else " ".join(content.split())[:160]

    listing = f"The listing is for: {row['product_title']} ({row['product_type']}). "
    what = ask("Name the main object or objects in this image in at most six words.", max_tokens=20)
    use = ask("What is the object in this image used for? Answer in one short sentence.", max_tokens=40)
    return {
        "part_of": ask(QUESTIONS["part_of"][0], YES_NO),
        "role_a": ask(listing + ROLE, ["product", "part", "set"]),
        "role_b": ask(listing + f"The image shows: {what}. It is used like this: {use} " + ROLE, ["product", "part", "set"]),
    }


def all_questions_in_one_request(model, row):
    """The 22 yes/no questions of the tree in a single request, with a structured reply. Returns {} when the reply is incomplete."""
    text = "Answer every question with yes or no.\n" + "\n".join(f"{key}: {QUESTIONS[key][0]}" for key in TREE_YES_NO)
    body = {"model": model, "temperature": 0, "max_tokens": 30 + 18 * len(TREE_YES_NO),
            "response_format": schema_format({key: {"type": "string", "enum": YES_NO} for key in TREE_YES_NO}),
            "messages": [{"role": "system", "content": SYSTEM}, image_message(_image(row), text)]}
    try:
        return json.loads(post_chat(body)["choices"][0]["message"]["content"])
    except ValueError:
        return {}


def run(strategy, model, rows, name, workers=8):
    """Apply a strategy to every image of rows that has no stored result yet. Results go to STORE/strategy_<name>.jsonl."""
    path = STORE / f"strategy_{name}.jsonl"
    done = {json.loads(line)["image_id"] for line in path.read_text().splitlines()} if path.exists() else set()
    todo = [row for row in rows.to_dict("records") if row["image_id"] not in done]

    def job(row):
        started = time.perf_counter()
        try:
            return {"image_id": row["image_id"], **strategy(model, row), "seconds": round(time.perf_counter() - started, 2)}
        except Exception:  # a failed image is tried again on the next run
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool, path.open("a") as stored:
        for record in pool.map(job, todo):
            if record:
                stored.write(json.dumps(record) + "\n")
                stored.flush()
    return path
