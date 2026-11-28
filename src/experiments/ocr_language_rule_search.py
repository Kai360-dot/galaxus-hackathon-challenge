"""OCR overrides for text_language and text_classification, tuned on out-of-fold train predictions, checked on the holdout."""
import json, sys, glob, itertools; sys.path.insert(0, "src")
import numpy as np, pandas as pd, common
from ensemble import macro_f1, scored_labels
from lingua import Language, LanguageDetectorBuilder
rows = pd.read_csv("data/train.csv", dtype=str)
ocr = {}
for f in glob.glob("results/ocr/train/ocr*.jsonl") + glob.glob("results/ocr/holdout/ocr.jsonl"):
    for line in open(f):
        r = json.loads(line); ocr[r["image_id"]] = r["boxes"]
detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
NAME = {Language.GERMAN: "German", Language.ENGLISH: "English", Language.FRENCH: "French", Language.ITALIAN: "Italian", Language.DUTCH: "Dutch"}
cache = {}
def reading(i, min_conf):
    key = (i, min_conf)
    if key not in cache:
        text = " ".join(b["text"] for b in ocr.get(i, []) if b["conf"] >= min_conf)
        letters = sum(ch.isalpha() for ch in text)
        lang, conf = None, 0.0
        if letters >= 8:
            values = detector.compute_language_confidence_values(text)
            if values:
                top = values[0]; lang, conf = NAME.get(top.language, "other"), top.value
        cache[key] = (letters, lang, conf)
    return cache[key]
L, T = common.CLASSES["text_language"], common.CLASSES["text_classification"]
def predict(probs, ids, min_conf, min_letters, min_lang_conf, cls_letters):
    lang, cls = [], []
    for i in ids:
        p = probs[i]; letters, l, c = reading(i, min_conf)
        a = L[int(np.argmax(p["text_language"]))]
        if min_letters and letters >= min_letters and l and c >= min_lang_conf:
            a = l
        lang.append(a)
        b = T[int(np.argmax(p["text_classification"]))]
        if cls_letters and b == "none" and letters >= cls_letters:
            q = p["text_classification"].copy(); q[T.index("none")] = -1; b = T[int(np.argmax(q))]
        cls.append(b)
    return np.array(lang), np.array(cls)
def evaluate(probs, part_rows, params):
    ids = list(part_rows.image_id)
    lang, cls = predict(probs, ids, *params)
    return macro_f1(scored_labels(part_rows, "text_language"), lang), macro_f1(scored_labels(part_rows, "text_classification"), cls)
oof = np.load("outputs/oof_base/oof_probabilities.npy", allow_pickle=True).item()
hold = np.load("outputs/oof_base/holdout_probabilities.npy", allow_pickle=True).item()
oof_rows = rows[rows.image_id.isin(oof)].reset_index(drop=True); hold_rows = rows[rows.image_id.isin(hold)].reset_index(drop=True)
base = (0.5, 0, 0, 0)
print("heads only        OOF lang %.3f cls %.3f | holdout lang %.3f cls %.3f" % (*evaluate(oof, oof_rows, base), *evaluate(hold, hold_rows, base)), flush=True)
best_l, best_c = (0, None), (0, None)
for mc, ml, lc in itertools.product([0.3, 0.5, 0.7], [10, 20, 40, 80], [0.5, 0.7, 0.9]):
    f, _ = evaluate(oof, oof_rows, (mc, ml, lc, 0))
    if f > best_l[0]: best_l = (f, (mc, ml, lc))
for mc, cl in itertools.product([0.3, 0.5, 0.7], [5, 10, 20, 40, 80]):
    _, f = evaluate(oof, oof_rows, (mc, 0, 0, cl))
    if f > best_c[0]: best_c = (f, (mc, cl))
print("best language rule on OOF:", best_l, flush=True)
print("best class rule on OOF:", best_c, flush=True)
mc, ml, lc = best_l[1]
print("language rule     OOF lang %.3f | holdout lang %.3f" % (evaluate(oof, oof_rows, (mc, ml, lc, 0))[0], evaluate(hold, hold_rows, (mc, ml, lc, 0))[0]))
mc, cl = best_c[1]
print("class rule        OOF cls %.3f | holdout cls %.3f" % (evaluate(oof, oof_rows, (mc, 0, 0, cl))[1], evaluate(hold, hold_rows, (mc, 0, 0, cl))[1]))

# Wider grid for the hard rule, and a soft blend: heads probabilities mixed with an OCR language distribution.
best_l = (0, None)
for mc, ml, lc in itertools.product([0.1, 0.2, 0.3], [3, 5, 8, 10, 15], [0.2, 0.3, 0.4, 0.5]):
    f, _ = evaluate(oof, oof_rows, (mc, ml, lc, 0))
    if f > best_l[0]: best_l = (f, (mc, ml, lc))
mc, ml, lc = best_l[1]
print("wider hard rule", best_l, "| holdout lang %.3f" % evaluate(hold, hold_rows, (mc, ml, lc, 0))[0], flush=True)
full = {}
def ocr_distribution(i, min_conf):
    key = (i, min_conf)
    if key not in full:
        text = " ".join(b["text"] for b in ocr.get(i, []) if b["conf"] >= min_conf)
        letters = sum(ch.isalpha() for ch in text)
        q = np.zeros(len(L))
        if letters < 3:
            q[L.index("none")] = 1
        else:
            for v in detector.compute_language_confidence_values(text):
                q[L.index(NAME.get(v.language, "other"))] += v.value
            q = q / q.sum() if q.sum() else q
        full[key] = q
    return full[key]
def soft(probs, part_rows, w, mc):
    pred = [L[int(np.argmax((1 - w) * probs[i]["text_language"] + w * ocr_distribution(i, mc)))] for i in part_rows.image_id]
    return macro_f1(scored_labels(part_rows, "text_language"), np.array(pred))
best_s = (0, None)
for w, mc in itertools.product([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.1, 0.3, 0.5]):
    f = soft(oof, oof_rows, w, mc)
    if f > best_s[0]: best_s = (f, (w, mc))
print("soft blend on OOF", best_s, "| holdout lang %.3f" % soft(hold, hold_rows, *best_s[1]), flush=True)
