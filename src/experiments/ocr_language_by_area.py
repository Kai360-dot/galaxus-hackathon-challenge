"""Targeted OCR language rule: only switch between languages (never from/to none), per-box languages weighted by area."""
import json, sys, glob, itertools; sys.path.insert(0, "src")
import numpy as np, pandas as pd, common
from ensemble import macro_f1, scored_labels, argmax_answers, write_submission
from pathlib import Path
from lingua import Language, LanguageDetectorBuilder
rows = pd.read_csv("data/train.csv", dtype=str)
ocr = {}
for f in glob.glob("results/ocr/*/ocr*.jsonl"):
    for line in open(f):
        r = json.loads(line); ocr[r["image_id"]] = r["boxes"]
det = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
NAME = {Language.GERMAN: "German", Language.ENGLISH: "English", Language.FRENCH: "French", Language.ITALIAN: "Italian", Language.DUTCH: "Dutch"}
ORDER = ["German", "English", "French", "Italian", "Dutch", "other"]
L = common.CLASSES["text_language"]
cache = {}
def area_language(i, min_conf, min_words, min_lang):
    key = (i, min_conf, min_words, min_lang)
    if key in cache: return cache[key]
    votes = {}
    for b in ocr.get(i, []):
        words = [w for w in b["text"].split() if sum(c.isalpha() for c in w) >= 2]
        if b["conf"] < min_conf or len(words) < min_words: continue
        values = det.compute_language_confidence_values(b["text"])
        if not values or values[0].value < min_lang: continue
        lang = NAME.get(values[0].language, "other")
        votes[lang] = votes.get(lang, 0) + max(0, b["x1"] - b["x0"]) * max(0, b["y1"] - b["y0"])
    result = None
    if votes:
        top = max(votes.values()); result = next(l for l in ORDER if votes.get(l, 0) >= 0.8 * top)  # near-tie: official order
    cache[key] = result
    return result
def predict(probs, ids, params):
    out = []
    for i in ids:
        a = L[int(np.argmax(probs[i]["text_language"]))]
        if params and a != "none":
            o = area_language(i, *params)
            if o: a = o
        out.append(a)
    return np.array(out)
oof = np.load("outputs/oof_mirror/oof_probabilities.npy", allow_pickle=True).item()
hold = np.load("outputs/oof_mirror/holdout_probabilities.npy", allow_pickle=True).item()
dev = np.load("outputs/oof_mirror/dev_probabilities.npy", allow_pickle=True).item()
oo = rows[rows.image_id.isin(oof)].reset_index(drop=True); ho = rows[rows.image_id.isin(hold)].reset_index(drop=True)
f = lambda p, r, params: macro_f1(scored_labels(r, "text_language"), predict(p, list(r.image_id), params))
print("heads only   OOF %.3f holdout %.3f" % (f(oof, oo, None), f(hold, ho, None)), flush=True)
best = (0, None)
for params in itertools.product([0.3, 0.5, 0.7], [1, 2, 3], [0.5, 0.7, 0.9]):
    v = f(oof, oo, params)
    if v > best[0]: best = (v, params)
print("best on OOF", best, "| holdout %.3f" % f(hold, ho, best[1]), flush=True)
base = predict(dev, list(dev), None); new = predict(dev, list(dev), best[1])
print("dev images changed:", int((base != new).sum()), "of", len(dev))
answers = argmax_answers(dev)
for k, i in enumerate(dev): answers[i]["text_language"] = new[k]
write_submission(Path("data"), answers, Path("outputs/submissions/dev_mirror_ocr2.csv"))
