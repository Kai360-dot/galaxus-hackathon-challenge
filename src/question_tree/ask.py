"""Asking the questions and storing the answers.

One request holds one question, with the image first. The server caches the image, so every further question on
the same image costs a fraction of the first one. Answers are appended to a jsonl file as (image_id, key, answer),
and a rerun only asks what is missing.
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .client import image_message, post_chat, schema_format
from .config import DATA_DIR, GROUP_ROBUST_QUESTIONS, WORKERS
from .images import data_url, prepared_image
from .questions import ASKED_TOGETHER, QUESTIONS, SYSTEM


def ask_alone(model, image_url, key, row):
    """The answer to one question about one image."""
    text, allowed = QUESTIONS[key]
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 16,
        "response_format": schema_format({"answer": {"type": "string", "enum": allowed}}),
        "messages": [{"role": "system", "content": SYSTEM}, image_message(image_url, text.format(**row) if "{" in text else text)],
    }
    return json.loads(post_chat(body)["choices"][0]["message"]["content"])["answer"]


def ask_together(model, image_url, keys):
    """The answers to several yes/no questions from one request. A question without a readable answer is left out.

    The reply is read with a pattern instead of a JSON parser: now and then the model stalls before the last
    question and fills the rest of its reply with blanks, and the answers it gave until then are still good.
    """
    text = "Answer every question with yes or no.\n" + "\n".join(f"{key}: {QUESTIONS[key][0]}" for key in keys)
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 40 + 14 * len(keys),
        "response_format": schema_format({key: {"type": "string", "enum": ["yes", "no"]} for key in keys}),
        "messages": [{"role": "system", "content": SYSTEM}, image_message(image_url, text)],
    }
    reply = post_chat(body)["choices"][0]["message"]["content"] or ""
    found = dict(re.findall(r'"([a-z_0-9]+)"\s*:\s*"(yes|no)"', reply))
    return {key: found[key] for key in keys if key in found}


def answer_image(model, row, keys, group=GROUP_ROBUST_QUESTIONS):
    """All answers for one image, as {key: answer}. A question that fails is missing from the result."""
    image_url = data_url(prepared_image(DATA_DIR / row["image_path"]))
    answers = {}
    together = [key for key in keys if key in ASKED_TOGETHER] if group else []
    if len(together) > 1:
        try:
            answers.update(ask_together(model, image_url, together))
        except Exception:  # the questions are then asked alone below
            pass
    for key in keys:
        if key not in answers:
            try:
                answers[key] = ask_alone(model, image_url, key, row)
            except Exception:  # a failed question is asked again on the next run
                pass
    return answers


def collect(model, rows, keys, path, group=GROUP_ROBUST_QUESTIONS, workers=WORKERS):
    """Ask the questions for every image of rows that has no stored answer yet. Returns images per second."""
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            record = json.loads(line)
            done.add((record["image_id"], record["key"]))

    def job(row):
        missing = [key for key in keys if (row["image_id"], key) not in done]
        if not missing:
            return []
        try:
            answers = answer_image(model, row, missing, group)
        except Exception:  # an unreadable image keeps no answers
            return []
        return [{"image_id": row["image_id"], "key": key, "answer": answer} for key, answer in answers.items()]

    records = rows.to_dict("records")
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool, path.open("a") as stored:
        for count, answered in enumerate(pool.map(job, records), start=1):
            for record in answered:
                stored.write(json.dumps(record) + "\n")
            stored.flush()
            if count % 200 == 0 or count == len(records):
                print(f"  {count} / {len(records)} images, {count / (time.perf_counter() - started):.2f} images/s", flush=True)
    return len(records) / (time.perf_counter() - started)


def read_answers(path):
    """The stored answers as a table: one row per image, one column per question."""
    records = [json.loads(line) for line in path.read_text().splitlines()]
    table = pd.DataFrame(records).drop_duplicates(["image_id", "key"], keep="last")
    return table.pivot(index="image_id", columns="key", values="answer")
