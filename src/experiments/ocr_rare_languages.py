"""OCR may only switch a predicted language to a rare one (Italian, Dutch, French, other), by area share and confidence."""
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
L = common.CLASSES["text_language"]; RARE = {"Italian", "Dutch", "French", "other"}
box_cache = {}
def box_langs(i):
    if i not in box_cache:
        out = []
        for b in ocr.get(i, []):
            words = [w for w in b["text"].split() if sum(c.isalpha() for c in w) >= 2]
            if len(words) < 2 or b["conf"] < 0.3: continue
            v = det.compute_language_confidence_values(b["text"])
            if v: out.append((NAME.get(v[0].language, "other"), v[0].value, max(0, b["x1"] - b["x0"]) * max(0, b["y1"] - b["y0"]), len(words)))
        box_cache[i] = out
    return box_cache[i]
def rare_vote(i, min_conf, min_share, min_words):
    boxes = [b for b in box_langs(i) if b[1] >= min_conf]
    total = sum(b[2] for b in box_langs(i)) or 1
    words = {}
    area = {}
    for lang, c, a, n in boxes:
        area[lang] = area.get(lang, 0) + a; words[lang] = words.get(lang, 0) + n
    cands = [(area[l], l) for l in area if l in RARE and area[l] / total >= min_share and words[l] >= min_words]
    return max(cands)[1] if cands else None
def predict(probs, ids, params):
    out = []
    for i in ids:
        a = L[int(np.argmax(probs[i]["text_language"]))]
        if params and a != "none" and a not in RARE:
            v = rare_vote(i, *params)
            if v: a = v
        out.append(a)
    return np.array(out)
oof = np.load("outputs/oof_mirror/oof_probabilities.npy", allow_pickle=True).item()
hold = np.load("outputs/oof_mirror/holdout_probabilities.npy", allow_pickle=True).item()
dev = np.load("outputs/oof_mirror/dev_probabilities.npy", allow_pickle=True).item()
oo = rows[rows.image_id.isin(oof)].reset_index(drop=True); ho = rows[rows.image_id.isin(hold)].reset_index(drop=True)
def report(p, r, params):
    y = scored_labels(r, "text_language"); k = y != None; pr = predict(p, list(r.image_id), params)
    f = {c: 2 * ((pr[k] == c) & (y[k] == c)).sum() / max(1, ((pr[k] == c).sum() + (y[k] == c).sum())) for c in ["Italian", "Dutch", "French", "other"]}
    return macro_f1(y, pr), f
print("heads only  OOF %.3f %s | holdout %.3f" % (*report(oof, oo, None), report(hold, ho, None)[0]), flush=True)
best = (0, None)
for params in itertools.product([0.7, 0.8, 0.9, 0.95], [0.3, 0.5, 0.7], [2, 4, 8]):
    v = report(oof, oo, params)[0]
    if v > best[0]: best = (v, params)
o = report(oof, oo, best[1]); h = report(hold, ho, best[1])
print("rare rule", best[1], "OOF %.3f %s | holdout %.3f %s" % (o[0], {k: round(v, 2) for k, v in o[1].items()}, h[0], {k: round(v, 2) for k, v in h[1].items()}))
base = predict(dev, list(dev), None); new = predict(dev, list(dev), best[1])
print("dev images changed:", int((base != new).sum()), dict(zip(*np.unique(new[base != new], return_counts=True))))
