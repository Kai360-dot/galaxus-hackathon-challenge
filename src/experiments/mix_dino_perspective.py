"""Mix the finetuned DINOv2 perspective into the heads (mirror run): weight from the validation products, check on holdout."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, common
from pathlib import Path
from heads import load_parts
from ensemble import score, argmax_answers, write_submission, macro_f1, scored_labels
run = Path(sys.argv[1]); tag = sys.argv[2]
parts = load_parts(Path("data")); metric = common.load_metric(Path("data"))
heads_oof = np.load("outputs/oof_mirror/oof_probabilities.npy", allow_pickle=True).item()
heads_hold = np.load("outputs/oof_mirror/holdout_probabilities.npy", allow_pickle=True).item()
heads_dev = np.load("outputs/oof_mirror/dev_probabilities.npy", allow_pickle=True).item()
d_val = np.load(run / "validation_probabilities.npy", allow_pickle=True).item()
d_hold = np.load(run / "holdout_probabilities.npy", allow_pickle=True).item()
d_dev = np.load(run / "dev_probabilities.npy", allow_pickle=True).item()
C = np.array(common.CLASSES["perspective"])
val_rows = parts["train"][parts["train"].image_id.isin(d_val)].reset_index(drop=True)
lab = scored_labels(val_rows, "perspective")
def f(w, hp, dp, ids, labels):
    return macro_f1(labels, C[np.array([((1 - w) * hp[i]["perspective"] + w * dp[i]).argmax() for i in ids])])
ws = np.round(np.arange(0, 1.01, 0.1), 2)
val_scores = {w: f(w, heads_oof, d_val, val_rows.image_id, lab) for w in ws}
print("validation perspective F1 by DINO weight:", " ".join(f"{w}:{v:.3f}" for w, v in val_scores.items()))
best = max(val_scores, key=val_scores.get)
hold_rows = parts["holdout"][parts["holdout"].image_id.isin(heads_hold)].reset_index(drop=True)
hl = scored_labels(hold_rows, "perspective")
print("holdout perspective F1 by DINO weight:", " ".join(f"{w}:{f(w, heads_hold, d_hold, hold_rows.image_id, hl):.3f}" for w in ws))
def mix(hp, dp, w):
    return {i: {**p, "perspective": (1 - w) * p["perspective"] + w * dp[i]} for i, p in hp.items()}
before = score(metric, hold_rows, heads_hold)["overall"]["f1_macro"]
after = score(metric, hold_rows, mix(heads_hold, d_hold, best))["overall"]["f1_macro"]
print(f"chosen weight {best} | holdout overall {before:.4f} -> {after:.4f}")
write_submission(Path("data"), argmax_answers(mix(heads_dev, d_dev, best)), Path(f"outputs/submissions/dev_mirror_dino{tag}.csv"))
