import sys, glob, os; sys.path.insert(0, "src")
import numpy as np, pandas as pd, common
from heads import load_parts
from ensemble import score
parts = load_parts("data" and __import__("pathlib").Path("data")); metric = common.load_metric(__import__("pathlib").Path("data"))
for d in sorted(glob.glob("outputs/oof_*")):
    if not os.path.exists(f"{d}/holdout_probabilities.npy"): print(os.path.basename(d), "not finished"); continue
    oof = np.load(f"{d}/oof_probabilities.npy", allow_pickle=True).item(); hold = np.load(f"{d}/holdout_probabilities.npy", allow_pickle=True).item()
    o = score(metric, parts["train"][parts["train"].image_id.isin(oof)].reset_index(drop=True), oof)
    h = score(metric, parts["holdout"][parts["holdout"].image_id.isin(hold)].reset_index(drop=True), hold)
    print(f"{os.path.basename(d):14} OOF {o['overall']['f1_macro']:.4f} | holdout {h['overall']['f1_macro']:.4f} | OOF per dim " + " ".join(f"{k[:4]} {o[k]['f1_macro']:.3f}" for k in common.DIMENSIONS))
