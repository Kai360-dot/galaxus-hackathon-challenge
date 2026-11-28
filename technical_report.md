# Technical report: Apertus sorts the shop, ShipHappens

The same text as [ShipHappens_Report.pdf](ShipHappens_Report.pdf). Its source is [docs/ShipHappens_Report.tex](docs/ShipHappens_Report.tex).

- **Challenge partner:** Digitec Galaxus
- **Challenge:** Hack Apertus x Galaxus: product image classification
- **Event:** On-site Zurich
- **Team:** ShipHappens: Kai Ruth, MAC, Hossam
- **Demo:** `link to video`

## 1. Summary

The task is to label every Galaxus product image along six dimensions with 41 classes in total. The score is the macro F1 per dimension, averaged over the six. Our final pipeline is one finetuned Apertus v1.5 8B that labels all six dimensions in a single forward pass, plus a small DINOv2 model that adds a second opinion on the perspective. It scores 0.852 on the public leaderboard (dev images). The frozen Apertus with trained heads that we started the day with scored 0.825, and the organisers’ Qwen 3.5 9B finetune scores 0.917.

## 2. Architecture

**Input.** The image is composited onto white, scaled to an area of about 512 x 512 pixels with its aspect ratio kept, and passed through a JPEG round trip. Apertus turns it into about 1,024 discrete image tokens (one per 16 x 16 pixels). The prompt holds one sentence that names the six dimensions, the product context (title, brand, product type, category) and the words “Classify this image.”

**Main model.** Apertus v1.5 8B runs up to decoder layer 24 of 32. A LoRA adapter (rank 32, about 60 M weights) sits on the attention and MLP projections of these 24 layers. Six MLP heads (one hidden layer of 512 units) read four pooled states after layer 24: the last token, the mean over the image tokens, and the difference of that mean between the left and right half and between the top and bottom half of the image. Each head outputs the class probabilities of one dimension. No text is generated.

**Helper.** DINOv2-large, finetuned on the perspective labels at 448 x 448 pixels, predicts the perspective from the image and its mirror image. The final perspective is the weighted mean of both models’ probabilities (Apertus 1, DINOv2 3). The other five dimensions come from Apertus alone.

**Target architecture.** The pipeline is deployable *on-premise* and *air-gapped*: it needs one GPU and local weight files, and calls no service at runtime. The same holds for a sovereign Swiss cloud with a GPU of 32 GB or more. Build-time dependencies are the Hugging Face Hub (weights of Apertus and DINOv2, our adapter), PyPI and the swiss-ai fork of `transformers`. Runtime dependencies: none.

## 3. Use of Apertus

- **Model:** `swiss-ai/Apertus-v1.5-8B`, revision `1519de0`, in bf16 without quantisation.

- **How it is used:** finetuning with LoRA and inference. Zero-shot experiments also used the 8B and 70B models through the Keera endpoint.

- **Where it runs:** local weights on one GPU, plain PyTorch with the swiss-ai fork of `transformers` (commit `3797303`) and PEFT.

**Training** (`src/lora_classifier.py`). Adapter and heads start untrained and learn together on the human labels of 11,276 train images. Loss: cross-entropy per dimension with class weights proportional to count^-0.5. Rows of rare classes are shown up to three times per epoch, the repeats at 448, 480 or 544 pixels. Images with a left or right perspective and no text are mirrored half of the time with the label swapped. The product context is dropped for 20% of the rows. AdamW, learning rate 2e-4 for the adapter and 5e-4 for the heads, 16 images per step, a cosine schedule over a fixed budget of 100 minutes: 1,767 steps, 2.27 epochs on one Nvidia H200. The image tokenizer runs in float32, outside the mixed-precision block, as the Apertus source asks; we did not measure the effect of this choice.

## 4. Data

Only the competition files from Kaggle: 11,876 labeled train images (600 images of 378 whole products held out for our own scoring), 5,261 dev images and 5,292 final images. The data is not redistributable. This repository and the published weights contain no images, csv files, labels or passphrase. Dev and final images were never labeled by hand.

## 5. Limitations

- **Text is the weak spot.** On the holdout, `text_classification` reaches 0.680 and `text_language` 0.799, against 0.842 and 0.916 for the Qwen reference notebook on the same images.

- **Rare classes.** Several classes have two or three holdout examples; their scores are unstable.

- **Small holdout.** With 600 images it misranked close variants (section 7). Final choices followed the public leaderboard.

- **One run.** One training run with one seed; no confidence intervals.

- **Speed.** 8.3 images per second on an H200, below the 28.7 images per second that the reference notebook reports for Qwen on an H100. We did not profile where the time goes.

- **Missing context.** The model was trained to cope without product context, but we did not score that case for the final pipeline.

## 6. Reproducibility

Hardware: one Nvidia H200 (141 GB) on Hugging Face Jobs. Seed 3407. Commands, from the repository root:

    JOB=lora_classifier_job.sh MINUTES=100 src/jobs/launch_job.sh <private-repo> h200 512  # 100 min training
    JOB=dino_perspective_job.sh src/jobs/launch_job.sh <private-repo> h200 448             # 5 min training
    FINAL_PASSPHRASE=... JOB=lora_classifier_job.sh MODE=predict src/jobs/launch_job.sh <private-repo> h200 512
    python src/blend.py --data-dir data --output submission.csv --source lora=<lora predictions> \
        --source dino:3=<dino predictions>

Commits on branch `feature/lora-classifier`, merged into `main`: `c978f82` (LoRA training), `56273ad` (prediction of the final images), `9a7d2e4` (DINOv2 job), `e9f7dee` (blend). Weights: <https://huggingface.co/KaiRuth/apertus-galaxus-lora-classifier>, revision `6f71705` (adapter, heads and the DINOv2 helper).

## 7. Galaxus challenge: results and resources

**Kaggle.** Team ShipHappens. Selected submission `final_B2_lora_plus_dino_weight3.csv`, public score 0.852.

| Method (mean macro F1) | Holdout, 600 images | Kaggle public |
|:---|:--:|:--:|
| One prompt for all six labels, zero-shot | 0.309(*) |  |
| Yes/no question tree, zero-shot | 0.586(*) |  |
| Frozen Apertus, six heads on layer 24 | 0.777 | 0.776 |
| \+ DINOv2 and pixel features, mirror head | 0.822 | 0.825 |
| \+ finetuned DINOv2 on perspective | 0.829 | 0.832 |
| Finetuned Apertus (LoRA + heads) alone | 0.808 | 0.833 |
| **Finetuned Apertus + DINOv2 on perspective (final)** | **0.826** | **0.852** |
| Finetuned Apertus averaged with the frozen pipeline (not selected) | 0.839 | 0.853 |

(*) on 573 of the 600 images; the other 27 were used to design the questions.


| Holdout, macro F1 | subject | background | perspective | individuals | text_class. | text_lang. | overall |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Final pipeline | 0.774 | 0.893 | 0.842 | 0.966 | 0.680 | 0.799 | 0.826 |
| Qwen reference notebook | 0.875 | 0.911 | 0.859 | 0.989 | 0.842 | 0.916 | 0.899 |


**What mattered.**

- **Finetuning Apertus itself.** Apertus 1.5 reads images as discrete tokens with its language layers, so only those layers can learn the task. Against the frozen model with heads, `text_language` rose from 0.711 to 0.799 on the holdout, and one model (0.833) matched the frozen pipeline with all its helpers (0.832).

- **A vision helper for the perspective.** Holdout perspective 0.736 to 0.842, public score 0.833 to 0.852.

- **Small questions in zero-shot.** One prompt for all labels gives almost one class for every image (0.287 on 500 images). One yes/no question per request and a decision tree reach 0.547, contour features for the perspective 0.609.

**What did not work.**

- The full labeling guideline as prompt (0.333 to 0.273), thinking mode (0.473 for six times the output tokens) and the 70B model (0.534 against 0.571 for 8B, about five times slower), all zero-shot on 200 images.

- An OCR rule for the text language (public 0.825 to 0.823) and class biases tuned on the holdout (0.804 to 0.774 on unseen products).

- LoRA on one question per dimension added nothing to the frozen heads (0.816 against 0.817).

- Averaging the frozen pipeline in as well: 0.853 against 0.852 on the public score, for a second Apertus pass per image.

- The holdout preferred a mix (0.853) that scored 0.845 on Kaggle, and rated at 0.839 the mix that scored 0.853.

**Models.** Runtime: Apertus v1.5 8B with our LoRA adapter and heads; `facebook/dinov2-large`, finetuned. Development only: Apertus v1.5 70B (zero-shot comparison), EasyOCR and Lingua (OCR rule), Qwen 3.5 9B numbers from the reference notebook. Claude Code was used as a coding assistant.

**Resources.** Prediction in batches of 32: Apertus 8.7 images per second on an H200, DINOv2 170 images per second, together 8.3 images per second. 100,000 images take 3.4 GPU hours, about 17 USD at 5 USD per hour. On an Nvidia L40S (48 GB) the Apertus part runs at 2.8 images per second with a peak of 29 GB. Training: 100 minutes for Apertus (peak 126 GB at batch size 8) and 5 minutes for DINOv2, both on one H200, about 11 USD in total. The 5,292 final images were predicted in 11 minutes.

**Label errors.** We prepared a list of 16 suspected label errors in train.csv for the hosts. Candidates were label cells where an out-of-fold model disagreed with at least 95% confidence; 144 of them were checked against the guideline with an AI assistant and spot-checked by us. The list is not in this repository.

## License

Creative Commons Attribution 4.0 (CC-BY-4.0). All HackApertus projects are open-sourced.
