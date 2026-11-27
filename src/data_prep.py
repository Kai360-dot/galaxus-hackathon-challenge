# Alpha compositing, mirroring, human-label
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

MAX_SIDE = 512

def opaque_rgb_on_white(image: Image.Image) -> Image.Image:
    if image.mode not in ("RGBA", "LA", "PA") and "transparency" not in image.info:
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB")

def process_one(args: tuple[Path, Path]) -> None:
    src_path, dst_path = args
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        rgb = opaque_rgb_on_white(img)
        w, h = rgb.size
        if max(w, h) > MAX_SIDE:
            scale = MAX_SIDE / max(w, h)
            new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
            rgb = rgb.resize(new_size, Image.Resampling.LANCZOS)
        # Save as high-quality JPEG (q=92) to avoid compression artifacts on text/white edges
        rgb.save(dst_path.with_suffix(".jpg"), format="JPEG", quality=92, optimize=True)