"""Pixel and contour features as helpers for the heads: no model, only measurements in the image.

The measurements come from the question tree on main (src/question_tree/pixels.py and contours.py):
- backdrop brightness at the four corners, also a few steps inside the border (the guideline: white is 250 or more),
- the silhouette, edge directions, brightness grid and shadows of the product (202 numbers, for perspective),
- a coarse colour grid and the corner colour (70 numbers).
They are copied here because the question_tree package needs its API settings and data folder at import.

Stored in the shard layout of the Apertus features (one layer, one pooling), so ensemble.py appends them like DINOv2.

    python src/pixel_features.py --data-dir ~/galaxus/data --features-dir results/pixels
"""

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import common
from heads import extraction_jobs, load_parts

SIZE = 160  # contours: every image squeezed to SIZE x SIZE
WHITE = 250
PATCH = 0.03
STEPS_INSIDE = (0.0, 0.01, 0.02, 0.04)


def _grid(values, cells=8):
    rows = np.linspace(0, values.shape[0], cells + 1).astype(int)
    columns = np.linspace(0, values.shape[1], cells + 1).astype(int)
    return [float(values[rows[i]: max(rows[i + 1], rows[i] + 1), columns[j]: max(columns[j + 1], columns[j] + 1)].mean())
            for i in range(cells) for j in range(cells)]


def shape_features(pixels):
    """202 numbers that describe the shape in a SIZE x SIZE x 3 image (from question_tree/contours.py)."""
    edge = 6
    corners = np.concatenate([pixels[:edge, :edge].reshape(-1, 3), pixels[:edge, -edge:].reshape(-1, 3),
                              pixels[-edge:, :edge].reshape(-1, 3), pixels[-edge:, -edge:].reshape(-1, 3)])
    backdrop = np.median(corners, axis=0)
    difference = np.abs(pixels - backdrop).max(axis=2)
    gray = pixels.mean(axis=2)
    mask = difference > 22
    ys, xs = np.nonzero(mask)
    if len(ys) < 30:
        return np.zeros(202, dtype=np.float32)
    top, bottom, left, right = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    box = mask[top:bottom, left:right].astype(np.float32)
    height, width = box.shape
    mirror_overlap = float((box * box[:, ::-1]).sum() / (np.maximum(box, box[:, ::-1]).sum() + 1e-9))
    flip_overlap = float((box * box[::-1]).sum() / (np.maximum(box, box[::-1]).sum() + 1e-9))
    outline = np.abs(np.diff(box, axis=0)).sum() + np.abs(np.diff(box, axis=1)).sum()
    summary = [width / height, box.mean(), mask.mean(), (xs.mean() - left) / width - 0.5,
               (ys.mean() - top) / height - 0.5, mirror_overlap, flip_overlap,
               float(outline * outline / (box.sum() + 1e-9)) / 100]
    inside = gray[top:bottom, left:right]
    gradient_y, gradient_x = np.gradient(inside)
    strength = np.hypot(gradient_x, gradient_y)
    direction = np.degrees(np.arctan2(gradient_y, gradient_x)) % 180

    def directions(part):
        histogram = np.histogram(direction[part], bins=12, range=(0, 180), weights=strength[part])[0]
        return list(histogram / (histogram.sum() + 1e-9))

    half_height, half_width = max(height // 2, 1), max(width // 2, 1)
    edges = (directions(np.s_[:, :]) + directions(np.s_[:half_height, :half_width])
             + directions(np.s_[:half_height, half_width:]) + directions(np.s_[half_height:, :half_width])
             + directions(np.s_[half_height:, half_width:]))
    soft = (difference > 4) & (difference <= 22) & (gray < backdrop.mean())
    shadow = [float(soft.mean()),
              float(soft[min(bottom, SIZE - 1):, :].mean()) if bottom < SIZE else 0.0,
              float(soft[:, :left].mean()) if left > 0 else 0.0,
              float(soft[:, right:].mean()) if right < SIZE else 0.0,
              float(soft[top + int(0.8 * height): bottom, left:right].mean())]
    return np.array(summary + _grid(box) + _grid(inside / 255) + edges + shadow + [float(corners.std()) / 50],
                    dtype=np.float32)


def colour_features(pixels):
    """70 numbers: mean red, green, blue and saturation on a 4 x 4 grid, corner colour, overall saturation."""
    saturation = (pixels.max(axis=2) - pixels.min(axis=2)) / 255
    cuts = np.linspace(0, SIZE, 5).astype(int)
    values = []
    for i in range(4):
        for j in range(4):
            cell = np.s_[cuts[i]: cuts[i + 1], cuts[j]: cuts[j + 1]]
            values += list(pixels[cell].reshape(-1, 3).mean(axis=0) / 255) + [float(saturation[cell].mean())]
    edge = 6
    corners = np.concatenate([pixels[:edge, :edge].reshape(-1, 3), pixels[:edge, -edge:].reshape(-1, 3),
                              pixels[-edge:, :edge].reshape(-1, 3), pixels[-edge:, -edge:].reshape(-1, 3)])
    values += list(np.median(corners, axis=0) / 255) + [float(saturation.mean()), float((saturation > 0.25).mean())]
    return np.array(values, dtype=np.float32)


def backdrop_features(rgb):
    """17 numbers: the four corner brightness medians at four steps inside the border (sorted), and white or not."""
    gray = np.asarray(rgb, dtype=np.float32).mean(axis=2)
    height, width = gray.shape
    side = max(2, int(PATCH * min(height, width)))
    values, white = [], False
    for step in STEPS_INSIDE:
        offset = int(round(step * min(height, width)))
        patches = [gray[offset: offset + side, offset: offset + side],
                   gray[offset: offset + side, width - offset - side: width - offset],
                   gray[height - offset - side: height - offset, offset: offset + side],
                   gray[height - offset - side: height - offset, width - offset - side: width - offset]]
        medians = sorted(float(np.median(p)) for p in patches)
        values += [m / 255 for m in medians]
        white = white or medians[2] >= WHITE
    return np.array(values + [float(white)], dtype=np.float32)


def measure(job):
    """All pixel features of one (image file, mirrored) job: 289 numbers."""
    image_file, mirrored = job
    try:
        with Image.open(image_file) as image:
            rgb = common.opaque_rgb_on_white(image)
    except Exception:  # an unreadable image counts as an empty white one
        rgb = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    if mirrored:
        rgb = ImageOps.mirror(rgb)
    pixels = np.asarray(rgb.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32)
    backdrop = backdrop_features(common.resize_to_area(rgb, 512))
    return np.concatenate([backdrop, shape_features(pixels), colour_features(pixels)])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, default=Path("results/pixels"))
    parser.add_argument("--parts", nargs="+", default=["holdout", "train", "dev"])
    parser.add_argument("--processes", type=int, default=12)
    arguments = parser.parse_args()
    parts = load_parts(arguments.data_dir)
    for part in arguments.parts:
        jobs = [(image_id, row, mirrored) for image_id, row, mirrored, with_context
                in extraction_jobs(part, parts[part], mirror=True, no_context_share=0.0) if with_context]
        started = time.time()
        with Pool(arguments.processes) as pool:
            features = np.stack(pool.map(measure, [(row["image_file"], m) for _, row, m in jobs], chunksize=40))
        part_dir = arguments.features_dir / part
        part_dir.mkdir(parents=True, exist_ok=True)
        np.savez(part_dir / "shard_0000.npz", image_ids=np.array([j[0] for j in jobs]),
                 mirrored=np.array([j[2] for j in jobs]), context=np.ones(len(jobs), dtype=bool),
                 features=features[:, None, None].astype(np.float32), layers=np.array([0]))
        print(f"{part}: {len(jobs)} images, {len(jobs) / (time.time() - started):.0f} images/s, "
              f"features {features.shape}", flush=True)


if __name__ == "__main__":
    main()
