import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, common
from ensemble import scored_labels
rows = pd.read_csv("data/train.csv", dtype=str)
oof = np.load("outputs/oof_mirror/oof_probabilities.npy", allow_pickle=True).item()
dval = np.load("runs/dino_perspective/validation_probabilities.npy", allow_pickle=True).item()
r = rows[rows.image_id.isin(oof)].reset_index(drop=True)
out = []
for d in common.DIMENSIONS:
    C = np.array(common.CLASSES[d]); y = scored_labels(r, d); keep = y != None
    pred = C[np.array([oof[i][d].argmax() for i in r.image_id])]
    y, pred = y[keep], pred[keep]
    for c in C:
        n = int((y == c).sum()); tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = n - tp
        if n: out.append((2 * tp / max(1, 2 * tp + fp + fn), d, c, n, int(fn), int(fp)))
for f, d, c, n, fn, fp in sorted(out)[:16]:
    print(f"{f:.2f}  {d:20} {c:22} n={n:5}  missed {fn:4}  wrongly predicted {fp:4}")
# perspective with DINO mix, on the validation products only
v = rows[rows.image_id.isin(dval)].reset_index(drop=True); C = np.array(common.CLASSES["perspective"]); y = scored_labels(v, "perspective"); k = y != None
pm = C[np.array([(0.4 * oof[i]["perspective"] + 0.6 * dval[i]).argmax() for i in v.image_id])][k]; ph = C[np.array([oof[i]["perspective"].argmax() for i in v.image_id])][k]; y = y[k]
print("\nperspective on the 724 validation images: heads -> heads+DINO")
for c in C:
    n = (y == c).sum()
    f = lambda p: 2 * ((p == c) & (y == c)).sum() / max(1, 2 * ((p == c) & (y == c)).sum() + ((p == c) & (y != c)).sum() + n - ((p == c) & (y == c)).sum())
    print(f"  {c:18} n={n:3}  {f(ph):.2f} -> {f(pm):.2f}")
