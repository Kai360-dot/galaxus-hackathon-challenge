import json, sys, glob, re, itertools; sys.path.insert(0, "src")
import numpy as np, pandas as pd, common
from ensemble import macro_f1, scored_labels
from ocr_features import NUTRITION, MEASURE
rows = pd.read_csv("data/train.csv", dtype=str)
ocr = {}
for f in glob.glob("results/ocr/train/ocr*.jsonl") + ["results/ocr/holdout/ocr.jsonl"]:
    for line in open(f):
        r = json.loads(line); ocr[r["image_id"]] = r["boxes"]
T = common.CLASSES["text_classification"]
def counts(i, mc):
    text = " ".join(b["text"] for b in ocr.get(i, []) if b["conf"] >= mc)
    return len(set(m.group(0).lower() for m in NUTRITION.finditer(text))), len(MEASURE.findall(text))
def run(probs, part_rows, nut, meas, p_tech, mc=0.3):
    pred = []
    for i in part_rows.image_id:
        p = probs[i]["text_classification"]; a = T[int(np.argmax(p))]
        n, m = counts(i, mc)
        if nut and n >= nut: a = "nutritional_info"
        elif meas and m >= meas and p[T.index("technical_specs")] >= p_tech and a != "nutritional_info": a = "technical_specs"
        pred.append(a)
    return macro_f1(scored_labels(part_rows, "text_classification"), np.array(pred))
oof = np.load("outputs/oof_mirror/oof_probabilities.npy", allow_pickle=True).item()
hold = np.load("outputs/oof_mirror/holdout_probabilities.npy", allow_pickle=True).item()
oo = rows[rows.image_id.isin(oof)].reset_index(drop=True); ho = rows[rows.image_id.isin(hold)].reset_index(drop=True)
print("heads only  OOF %.3f holdout %.3f" % (run(oof, oo, 0, 0, 1), run(hold, ho, 0, 0, 1)))
best = (0, None)
for nut, meas, pt in itertools.product([0, 2, 3, 4, 5], [0, 2, 3, 5, 8], [0.0, 0.05, 0.1, 0.2]):
    f = run(oof, oo, nut, meas, pt)
    if f > best[0]: best = (f, (nut, meas, pt))
print("best on OOF", best, "| holdout %.3f" % run(hold, ho, *best[1]))
for nut in [2, 3, 4, 5]:
    print(f"nutrition only >= {nut}: OOF %.3f holdout %.3f" % (run(oof, oo, nut, 0, 1), run(hold, ho, nut, 0, 1)))
