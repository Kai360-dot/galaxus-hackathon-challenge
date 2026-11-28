"""Local HTML page: train images the model gets wrong with high confidence (out-of-fold), per dimension."""
import base64, io, sys, html
from collections import Counter
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from PIL import Image
import common
rows = pd.read_csv("data/train.csv", dtype=str).set_index("image_id")
oof = np.load("outputs/oof_base/oof_probabilities.npy", allow_pickle=True).item()
PER_DIM = 18
def thumb(path):
    with Image.open(path) as im:
        im = common.opaque_rgb_on_white(im); im.thumbnail((240, 240)); b = io.BytesIO(); im.save(b, "JPEG", quality=80)
    return base64.b64encode(b.getvalue()).decode()
parts = ["<!doctype html><meta charset=utf-8><title>Confident errors (out-of-fold)</title><style>body{font-family:system-ui;margin:24px;color:#111}h2{margin-top:36px}"
         ".g{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px}.c{border:1px solid #ddd;border-radius:8px;padding:8px;font-size:13px}"
         ".c img{width:100%;height:200px;object-fit:contain;background:#f5f5f5}.l{color:#0a7d32}.p{color:#c0392b;font-weight:600}.m{color:#777}"
         "table{border-collapse:collapse;margin:8px 0}td{padding:2px 10px;border-bottom:1px solid #eee}</style>",
         "<h1>Images the model gets wrong with high confidence</h1><p class=m>Out-of-fold: each image was predicted by heads that never saw it (5 folds, 5 seeds). "
         "Only scored cells (human label, relevant). Sorted by how sure the model is of its wrong answer: at the top are likely label errors or genuinely ambiguous images.</p>"]
for d in common.DIMENSIONS:
    names = common.CLASSES[d]
    cand, pairs, total = [], Counter(), 0
    for i, p in oof.items():
        r = rows.loc[i]
        if r[f"{d}_source"] != "human" or str(r[f"{d}_relevant"]).lower() != "true":
            continue
        total += 1
        k = int(np.argmax(p[d])); pred = names[k]
        if pred != r[d]:
            pairs[(r[d], pred)] += 1
            cand.append((float(p[d][k]), float(p[d][names.index(r[d])]), i, r[d], pred))
    cand.sort(reverse=True)
    parts.append(f"<h2>{d}: {len(cand)} wrong of {total} ({100*len(cand)/total:.1f}%)</h2><table><tr><td><b>label</b></td><td><b>predicted</b></td><td><b>images</b></td></tr>"
                 + "".join(f"<tr><td>{a}</td><td>{b}</td><td>{n}</td></tr>" for (a, b), n in pairs.most_common(8)) + "</table><div class=g>")
    for conf, p_label, i, label, pred in cand[:PER_DIM]:
        r = rows.loc[i]
        parts.append(f"<div class=c><img src='data:image/jpeg;base64,{thumb('data/' + r['image_path'])}'><div>label <span class=l>{label}</span> ({p_label:.0%})</div>"
                     f"<div>model <span class=p>{pred}</span> ({conf:.0%})</div><div class=m>{html.escape(str(r['product_title'])[:60])} · {i}</div></div>")
    parts.append("</div>")
open("/tmp/confident_errors.html", "w").write("".join(parts))
print("written")
