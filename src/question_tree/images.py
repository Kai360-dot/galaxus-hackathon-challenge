"""Image preparation, the same for every request."""

import base64
import io

from PIL import Image

from .config import IMAGE_SIZE, JPEG_QUALITY


def opaque_rgb_on_white(image):
    """RGB, with any transparency composited onto white (the function of the competition README)."""
    if image.mode not in ("RGBA", "LA", "PA") and "transparency" not in image.info:
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB")


def load_rgb(image_path):
    """The image as the labelers saw it: opaque, on white."""
    with Image.open(image_path) as image:
        return opaque_rgb_on_white(image)


def prepared_image(image_path):
    """The image as the model gets it: on white, about IMAGE_SIZE squared pixels, aspect ratio kept."""
    rgb = load_rgb(image_path)
    scale = IMAGE_SIZE / (rgb.width * rgb.height) ** 0.5
    scale = min(scale, 4 * IMAGE_SIZE / max(rgb.size))  # a thin strip would get a very long side otherwise
    size = (max(32, round(rgb.width * scale)), max(32, round(rgb.height * scale)))
    return rgb.resize(size, Image.LANCZOS)


def data_url(image):
    """A PIL image as a base64 JPEG data URL."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
