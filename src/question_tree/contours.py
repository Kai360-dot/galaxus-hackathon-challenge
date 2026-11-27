"""Perspective from the shape of the product: silhouette, edge directions, brightness and shadows.

Asking Apertus from which side a product is seen barely beats a constant answer. The view does show in the
outline, though: a product seen from the left is the mirror image of the same product seen from the right.
So every image is reduced to a few hundred numbers and a small tree ensemble learns the perspective from them.
No language model is involved.
"""

from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier

from .config import DATA_DIR, SEED, STORE
from .images import load_rgb

SIZE = 160  # every image is squeezed to SIZE x SIZE before measuring
CACHE = STORE / "contours.npz"
MIRRORED_CLASS = {
    "front_angle_left": "front_angle_right", "front_angle_right": "front_angle_left",
    "back_angle_left": "back_angle_right", "back_angle_right": "back_angle_left",
    "left_side": "right_side", "right_side": "left_side",
}
FEATURE_GROUPS = (
    ["summary"] * 8 + ["silhouette"] * 64 + ["brightness"] * 64 + ["edge directions"] * 60 + ["shadow"] * 5 + ["summary"]
)


def _grid(values, cells=8):
    """Mean of values in every cell of a cells x cells grid."""
    rows = np.linspace(0, values.shape[0], cells + 1).astype(int)
    columns = np.linspace(0, values.shape[1], cells + 1).astype(int)
    return [
        float(values[rows[i] : max(rows[i + 1], rows[i] + 1), columns[j] : max(columns[j + 1], columns[j] + 1)].mean())
        for i in range(cells)
        for j in range(cells)
    ]


def shape_features(pixels):
    """202 numbers that describe the shape in a SIZE x SIZE x 3 image. Colour is not used."""
    edge = 6
    corners = np.concatenate([
        pixels[:edge, :edge].reshape(-1, 3), pixels[:edge, -edge:].reshape(-1, 3),
        pixels[-edge:, :edge].reshape(-1, 3), pixels[-edge:, -edge:].reshape(-1, 3),
    ])
    backdrop = np.median(corners, axis=0)
    difference = np.abs(pixels - backdrop).max(axis=2)
    gray = pixels.mean(axis=2)
    mask = difference > 22  # the silhouette: everything that differs from the backdrop
    ys, xs = np.nonzero(mask)
    if len(ys) < 30:
        return np.zeros(len(FEATURE_GROUPS), dtype=np.float32)
    top, bottom, left, right = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    box = mask[top:bottom, left:right].astype(np.float32)
    height, width = box.shape
    mirror_overlap = float((box * box[:, ::-1]).sum() / (np.maximum(box, box[:, ::-1]).sum() + 1e-9))
    flip_overlap = float((box * box[::-1]).sum() / (np.maximum(box, box[::-1]).sum() + 1e-9))
    outline = np.abs(np.diff(box, axis=0)).sum() + np.abs(np.diff(box, axis=1)).sum()
    summary = [
        width / height, box.mean(), mask.mean(), (xs.mean() - left) / width - 0.5, (ys.mean() - top) / height - 0.5,
        mirror_overlap, flip_overlap, float(outline * outline / (box.sum() + 1e-9)) / 100,
    ]
    # contour lines: which directions do the edges run in, for the whole box and for its four quarters
    inside = gray[top:bottom, left:right]
    gradient_y, gradient_x = np.gradient(inside)
    strength = np.hypot(gradient_x, gradient_y)
    direction = np.degrees(np.arctan2(gradient_y, gradient_x)) % 180

    def directions(part):
        histogram = np.histogram(direction[part], bins=12, range=(0, 180), weights=strength[part])[0]
        return list(histogram / (histogram.sum() + 1e-9))

    half_height, half_width = max(height // 2, 1), max(width // 2, 1)
    edges = (
        directions(np.s_[:, :]) + directions(np.s_[:half_height, :half_width]) + directions(np.s_[:half_height, half_width:])
        + directions(np.s_[half_height:, :half_width]) + directions(np.s_[half_height:, half_width:])
    )
    # shadow: soft pixels a little darker than the backdrop, and where they lie
    soft = (difference > 4) & (difference <= 22) & (gray < backdrop.mean())
    shadow = [
        float(soft.mean()),
        float(soft[min(bottom, SIZE - 1) :, :].mean()) if bottom < SIZE else 0.0,
        float(soft[:, :left].mean()) if left > 0 else 0.0,
        float(soft[:, right:].mean()) if right < SIZE else 0.0,
        float(soft[top + int(0.8 * height) : bottom, left:right].mean()),
    ]
    return np.array(summary + _grid(box) + _grid(inside / 255) + edges + shadow + [float(corners.std()) / 50], dtype=np.float32)


def colour_features(pixels):
    """Coarse colour: mean red, green, blue and saturation on a 4 x 4 grid, and the corner colour."""
    saturation = (pixels.max(axis=2) - pixels.min(axis=2)) / 255
    cuts = np.linspace(0, SIZE, 5).astype(int)
    values = []
    for i in range(4):
        for j in range(4):
            cell = np.s_[cuts[i] : cuts[i + 1], cuts[j] : cuts[j + 1]]
            values += list(pixels[cell].reshape(-1, 3).mean(axis=0) / 255) + [float(saturation[cell].mean())]
    edge = 6
    corners = np.concatenate([
        pixels[:edge, :edge].reshape(-1, 3), pixels[:edge, -edge:].reshape(-1, 3),
        pixels[-edge:, :edge].reshape(-1, 3), pixels[-edge:, -edge:].reshape(-1, 3),
    ])
    values += list(np.median(corners, axis=0) / 255) + [float(saturation.mean()), float((saturation > 0.25).mean())]
    return np.array(values, dtype=np.float32)


def _measure(image_path):
    """The features of one image: its shape, the shape of its mirror image, and its colour."""
    try:
        pixels = np.asarray(load_rgb(DATA_DIR / image_path).resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32)
    except Exception:  # an unreadable image counts as an empty white one
        pixels = np.full((SIZE, SIZE, 3), 255, dtype=np.float32)
    return shape_features(pixels), shape_features(pixels[:, ::-1]), colour_features(pixels)


def measure(rows, processes=8):
    """Features for every image of rows. Measured once, then read from the cache file."""
    cached = {}
    if CACHE.exists():
        stored = np.load(CACHE, allow_pickle=True)
        cached = {image_id: index for index, image_id in enumerate(stored["image_id"])}
    missing = rows[~rows["image_id"].isin(cached)]
    if len(missing):
        with Pool(processes) as pool:
            measured = pool.map(_measure, list(missing["image_path"]), chunksize=40)
        new = {
            "image_id": missing["image_id"].to_numpy(),
            "shape": np.stack([m[0] for m in measured]),
            "mirrored": np.stack([m[1] for m in measured]),
            "colour": np.stack([m[2] for m in measured]),
        }
        if cached:
            new = {key: np.concatenate([stored[key], new[key]]) for key in new}
        np.savez(CACHE, **new)
        stored = np.load(CACHE, allow_pickle=True)
        cached = {image_id: index for index, image_id in enumerate(stored["image_id"])}
    order = [cached[image_id] for image_id in rows["image_id"]]
    return {key: stored[key][order] for key in ("shape", "mirrored", "colour")}


class PerspectiveModel:
    """Predicts the perspective from the shape features and the product subcategory."""

    def __init__(self, use_mirrored=True, use_subcategory=True, trees=400):
        self.use_mirrored, self.use_subcategory = use_mirrored, use_subcategory
        self.forest = ExtraTreesClassifier(n_estimators=trees, min_samples_leaf=2, n_jobs=-1, random_state=SEED)

    def _inputs(self, shapes, rows):
        if not self.use_subcategory:
            return shapes
        subcategory = pd.get_dummies(rows["subcategory"]).reindex(columns=self.subcategories, fill_value=0)
        return np.hstack([shapes, subcategory.to_numpy(dtype=np.float32)])

    def fit(self, rows, features):
        """rows must carry a perspective label. A mirrored image is a second example with left and right swapped."""
        self.subcategories = sorted(rows["subcategory"].dropna().unique())
        inputs, labels = self._inputs(features["shape"], rows), rows["perspective"].to_numpy()
        if self.use_mirrored:
            mirrored_labels = np.array([MIRRORED_CLASS.get(label, label) for label in labels])
            inputs = np.vstack([inputs, self._inputs(features["mirrored"], rows)])
            labels = np.concatenate([labels, mirrored_labels])
        self.forest.fit(inputs, labels)
        return self

    def predict(self, rows, features):
        return self.forest.predict(self._inputs(features["shape"], rows))

    def importance_by_group(self):
        """Share of the decisions that each group of features carries."""
        groups = FEATURE_GROUPS + ["subcategory"] * (len(self.forest.feature_importances_) - len(FEATURE_GROUPS))
        return pd.Series(self.forest.feature_importances_).groupby(groups).sum().sort_values(ascending=False)
