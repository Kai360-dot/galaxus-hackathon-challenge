# Experiments

One-off analysis scripts run from the repository root on the machine with the data (`python src/experiments/<script>.py`).
They read the outputs of `src/oof.py` (outputs/oof_*) and `src/dino_finetune.py` (runs/dino_perspective*) and the raw
OCR of `src/ocr_features.py` (results/ocr). No data, labels or images are stored here.

| Script | Question | Result (out-of-fold on 11k train images / 600 holdout) |
| --- | --- | --- |
| score_oof_runs.py | macro F1 of every oof.py variant | base 0.801 / 0.818; mirror perspective head 0.809 / 0.822; mixup, inverse-prevalence weights, class-balanced sampling, noise, product mean, context-only: no gain |
| per_class_f1.py | weakest classes | Italian, Dutch, other language; product_grouping, accessories, multiview; certification_seal, other text |
| confident_errors_page.py | images the model is surest about and wrong (local HTML page) | many ambiguous or likely mislabeled images |
| ocr_language_rule_search.py | OCR + Lingua overriding text_language | +0.04 out-of-fold, but Kaggle dev 0.8249 -> 0.8202 |
| ocr_language_by_area.py | only language <-> language, per box by area | +0.06 out-of-fold, Kaggle dev 0.8249 -> 0.8231 |
| ocr_rare_languages.py | only switch to Italian, Dutch, French, other | +0.04 out-of-fold, holdout 0.711 -> 0.855, 35 dev images change |
| ocr_text_class_keywords.py | OCR keywords for text_classification | no gain |
| mix_dino_perspective.py | finetuned DINOv2 perspective mixed into the heads | holdout perspective 0.815 -> 0.858, overall 0.822 -> 0.829 |

Note on the rules: Apertus must perform the classification. Mixing a separate DINOv2 or OCR decision into the output
is a grey zone; the compliant form feeds these signals as inputs to the Apertus heads.
