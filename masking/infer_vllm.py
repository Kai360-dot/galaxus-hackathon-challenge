import argparse
import json
import sys
from pathlib import Path
import pandas as pd
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from vllm.lora.request import LoRARequest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from metric import DIMENSIONS, VALID_CLASSES

MODEL_ID = "swiss-ai/Apertus-v1.5-8B"
SYSTEM_PROMPT = (
    "Classify this e-commerce product image into JSON with exact keys: "
    "subject, background, perspective, individuals, text_classification, text_language."
)

JSON_SCHEMA = {
    "type": "object",
    "properties": {dim: {"type": "string", "enum": sorted(VALID_CLASSES[dim])} for dim in DIMENSIONS},
    "required": list(DIMENSIONS),
    "additionalProperties": False,
}

SPLIT_CONFIG = {
    "synthetic": ("data/synthetic_test.csv", "data/synthetic_images", "data/synthetic_preds_raw.csv"),
    "dev": ("data/dev.csv", "data/dev_images", "data/dev_preds_raw.csv"),
    "final": ("data/final/test.csv", "data/final/images", "data/final_preds_raw.csv"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=list(SPLIT_CONFIG.keys()), default="synthetic")
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--img-dir", default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args, _ = parser.parse_known_args()

    default_csv, default_img, default_out = SPLIT_CONFIG[args.split]
    csv_path = args.csv_path or default_csv
    img_dir = Path(args.img_dir or default_img)
    out_csv = args.out_csv or default_out

    df = pd.read_csv(csv_path)

    llm = LLM(
        model=MODEL_ID,
        enable_lora=args.lora_path is not None,
        max_lora_rank=64,
        max_model_len=4096,
    )
    lora_req = LoRARequest("galaxus_lora", 1, args.lora_path) if args.lora_path else None

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        structured_outputs=StructuredOutputsParams(json=JSON_SCHEMA),
    )

    rows = []
    for start in range(0, len(df), args.batch_size):
        chunk = df.iloc[start : start + args.batch_size]
        prompts = []
        valid_ids = []
        for _, row in chunk.iterrows():
            img_id = str(row["image_id"])
            matches = list(img_dir.glob(f"{img_id}.*"))
            if not matches:
                rows.append({
                    "image_id": img_id,
                    "subject": "full_product",
                    "background": "white",
                    "perspective": "front_head_on",
                    "individuals": "none",
                    "text_classification": "none",
                    "text_language": "none",
                })
                continue
            im = Image.open(matches[0]).convert("RGB")
            prompts.append({
                "prompt": f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n<|image|>\nClassify this image.\n<|assistant|>\n",
                "multi_modal_data": {"image": im},
            })
            valid_ids.append(img_id)

        if prompts:
            outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
            for img_id, out in zip(valid_ids, outputs):
                parsed = json.loads(out.outputs[0].text)
                parsed["image_id"] = img_id
                rows.append(parsed)

    out_df = df[["image_id"]].merge(pd.DataFrame(rows)[["image_id", *DIMENSIONS]], on="image_id", how="left")
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(out_df)} schema-constrained predictions to {out_csv}")


if __name__ == "__main__":
    main()