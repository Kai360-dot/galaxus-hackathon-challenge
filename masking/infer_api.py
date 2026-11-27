import argparse
import base64
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from metric import DIMENSIONS, VALID_CLASSES

SYSTEM_PROMPT = (
    "You are a product image classifier for Galaxus. Classify the image into JSON with exact keys: "
    "subject, background, perspective, individuals, text_classification, text_language.\n"
    + "\n".join(f"- {d}: {sorted(VALID_CLASSES[d])}" for d in DIMENSIONS)
    + "\nRespond ONLY with valid JSON using exact allowed class names."
)

SPLIT_CONFIG = {
    "synthetic": ("data/synthetic_test.csv", "data/synthetic_images", "data/synthetic_preds_raw.csv"),
    "dev": ("data/dev.csv", "data/dev_images", "data/dev_preds_raw.csv"),
    "final": ("data/final/test.csv", "data/final/images", "data/final_preds_raw.csv"),
}

def encode_image(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def classify_one(client: OpenAI, model: str, img_id: str, img_dir: Path) -> dict:
    matches = list(img_dir.glob(f"{img_id}.*"))
    if not matches:
        return {"image_id": img_id}
    data_url = encode_image(matches[0])
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=128,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "Classify this product image."},
                ],
            },
        ],
    )
    parsed = json.loads(resp.choices[0].message.content)
    parsed["image_id"] = img_id
    return parsed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=list(SPLIT_CONFIG.keys()), default="synthetic")
    parser.add_argument("--model", default=os.getenv("APERTUS_MODEL", "swiss-ai/Apertus-v1.5-8B"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    client = OpenAI(
        api_key=os.environ["APERTUS_API_KEY"],
        base_url=os.environ["APERTUS_BASE_URL"],
    )
    csv_path, img_dir_str, out_csv = SPLIT_CONFIG[args.split]
    df = pd.read_csv(csv_path)
    img_dir = Path(img_dir_str)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(classify_one, client, args.model, str(r["image_id"]), img_dir) for _, r in df.iterrows()]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            results.append(fut.result())

    out_df = df[["image_id"]].merge(pd.DataFrame(results), on="image_id", how="left")
    out_df.to_csv(out_csv, index=False)
    print(f"Saved {len(out_df)} API predictions to {out_csv}")

if __name__ == "__main__":
    main()