"""DINOv2 image features as a helper for the heads (geometry for perspective, pixels for background).

Galaxus suggests self-supervised encoders like DINOv2 as open-weights helpers: "features learned without text may keep
the geometry perspective needs". Per image we keep the CLS token and the mean over the patch tokens of
facebook/dinov2-large (Apache-2.0), in the same shard layout as the Apertus features, so heads.py and ensemble.py read
them as they are (one "layer" 0, poolings last = CLS and mean = patches).

    python src/dino.py --data-dir ~/galaxus/data --features-dir results/dino_224 --parts holdout train dev
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm.auto import tqdm

import common
from heads import extraction_jobs, load_parts

MODEL = "facebook/dinov2-large"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class Images(torch.utils.data.Dataset):
    """Each image on white (transparency), at its native aspect ratio, scaled to about size x size pixels.

    No padding and no crop: white padding would look like a white background, and a crop would hide cut-off edges.
    DINOv2 interpolates its position embeddings, so every image keeps its own shape (sides a multiple of 14).
    """

    def __init__(self, jobs, size):
        self.jobs, self.size = jobs, size

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, index):
        _, row, mirrored = self.jobs[index]
        with Image.open(row["image_file"]) as image:
            rgb = common.opaque_rgb_on_white(image)
        if mirrored:
            rgb = ImageOps.mirror(rgb)
        resized = common.resize_to_area(rgb, self.size, multiple=14, max_side_factor=4)
        pixels = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return (pixels - MEAN) / STD


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, default=Path("results/dino_224"))
    parser.add_argument("--parts", nargs="+", default=["holdout", "train", "dev"])
    parser.add_argument("--size", type=int, default=224, help="images scaled to about size x size pixels")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    arguments = parser.parse_args()

    from transformers import AutoModel

    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    parts = load_parts(arguments.data_dir)
    for part in arguments.parts:
        rows = parts[part] if arguments.limit == 0 else parts[part].head(arguments.limit)
        # Same images as the Apertus features: mirrored copies for train; the context copies need no own features.
        jobs = [(image_id, row, mirrored) for image_id, row, mirrored, with_context
                in extraction_jobs(part, rows, mirror=True, no_context_share=0.0) if with_context]
        # One image per batch: every image keeps its own aspect ratio.
        loader = torch.utils.data.DataLoader(Images(jobs, arguments.size), batch_size=None,
                                             num_workers=arguments.workers)
        features, started = [], time.time()
        with torch.inference_mode():
            for pixels in tqdm(loader, desc=part, mininterval=30):
                hidden = model(pixel_values=pixels[None].cuda().half()).last_hidden_state
                cls, patches = hidden[:, 0], hidden[:, 1:].mean(dim=1)
                features.append(torch.stack([cls, patches], dim=1).float().cpu().numpy().astype(np.float16))
        features = np.concatenate(features)[:, None]  # (images, 1 layer, 2 poolings, 1024)
        part_dir = arguments.features_dir / part
        part_dir.mkdir(parents=True, exist_ok=True)
        np.savez(part_dir / "shard_0000.npz", image_ids=np.array([j[0] for j in jobs]),
                 mirrored=np.array([j[2] for j in jobs]), context=np.ones(len(jobs), dtype=bool),
                 features=features, layers=np.array([0]))
        print(f"{part}: {len(jobs)} images, {len(jobs) / (time.time() - started):.0f} images/s, "
              f"features {features.shape}", flush=True)


if __name__ == "__main__":
    main()
