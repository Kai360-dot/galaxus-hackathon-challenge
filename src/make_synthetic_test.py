import os
import random
from pathlib import Path
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)

TRAIN_CSV = "data/train.csv"
TRAIN_IMG_DIR = Path("data/train_images")
OUT_CSV = "data/synthetic_test.csv"
OUT_IMG_DIR = Path("data/synthetic_images")

def perturb_image(img: Image.Image) -> Image.Image:
    # Keep aspect ratio & orientation intact so left/right perspective labels remain valid
    w, h = img.size
    crop_frac = random.uniform(0.92, 0.98)
    nw, nh = int(w * crop_frac), int(h * crop_frac)
    left = random.randint(0, max(0, w - nw))
    top = random.randint(0, max(0, h - nh))
    img = img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.Resampling.LANCZOS)

    # Mild lighting & sensor variation
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.88, 1.12))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.90, 1.12))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.92, 1.08))

    if random.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

    return img

def main():
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TRAIN_CSV)

    # Create a composite stratification key across subject + text_classification to protect rare classes
    strat_key = df["subject"].fillna("none") + "_" + df["text_classification"].fillna("none")
    counts = strat_key.value_counts()
    strat_key = strat_key.where(strat_key.map(counts) >= 2, "rare_combo")

    _, synth_df = train_test_split(df, test_size=0.15, random_state=SEED, stratify=strat_key)
    synth_df = synth_df.copy().reset_index(drop=True)

    valid_rows = []
    for _, row in synth_df.iterrows():
        img_id = str(row["image_id"])
        # Match .jpg or .png in data/train_images
        matches = list(TRAIN_IMG_DIR.glob(f"{img_id}.*"))
        if not matches:
            continue
        src_path = matches[0]
        ext = src_path.suffix.lower()
        new_id = f"synth_{img_id}"
        dst_path = OUT_IMG_DIR / f"{new_id}{ext}"

        with Image.open(src_path) as im:
            im = im.convert("RGB")
            aug = perturb_image(im)
            # Simulate realistic web compression artifacts
            aug.save(dst_path, quality=random.randint(72, 92))

        row["image_id"] = new_id
        valid_rows.append(row)

    out_df = pd.DataFrame(valid_rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(out_df)} synthetic test samples to {OUT_CSV} and {OUT_IMG_DIR}/")

if __name__ == "__main__":
    main()