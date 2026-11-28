"""Live demo: photograph a product with a phone, watch the model label and sort its images on a big screen.

One page, two views: on a phone it opens the camera and sends photos; on a large screen it shows the shop gallery
the labels produce (hero image first) and a board that sorts every photo into the columns of one dimension.

The model is the one of lora_heads.py: Apertus up to layer 24 (with a LoRA adapter, or the plain model), DINOv2 at
448 px, pixel and contour measures, six heads. Everything runs in this process on one GPU.

    python src/demo/app.py --heads runs/lora_heads/init_heads.pt --quantization 4bit          # test, 12 GB GPU
    python src/demo/app.py --checkpoint runs/lora_heads/step_150 --quantization none          # bf16 on a rented GPU

Then open http://<host>:7860 on a laptop (board) and on a phone (camera).
"""

import argparse
import io
import queue
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, Response  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

import common  # noqa: E402
from lora_heads import Heads, apertus_features, load_apertus, prompt_text  # noqa: E402
from pixel_features import measure  # noqa: E402

HERE = Path(__file__).resolve().parent

# Gallery order, as the shop would sort a product's images: what the customer buys first, logos and placeholders last.
SUBJECT_ORDER = ["full_product", "package_and_product", "multiview", "detail_closeup", "delivery_scope",
                 "product_grouping", "multipack_trade_unit", "accessories", "packaging", "infographic",
                 "energy_label", "logo", "placeholder"]
HIDDEN = {"placeholder", "logo"}
PERSPECTIVE_ORDER = ["front_angle_left", "front_angle_right", "front_head_on", "top_down", "left_side",
                     "right_side", "back_angle_left", "back_angle_right", "back_head_on", "bottom_up"]


class Model:
    """Apertus (to layer 24) + DINOv2 448 + pixel measures + six heads, for one image at a time."""

    def __init__(self, arguments):
        from transformers import AutoModel

        adapter = arguments.checkpoint / "adapter" if arguments.checkpoint else None
        heads_file = arguments.checkpoint / "heads.pt" if arguments.checkpoint else arguments.heads
        self.processor, self.apertus, self.captured = load_apertus(arguments, adapter=adapter)
        self.apertus.eval()
        self.device = next(self.apertus.parameters()).device
        self.heads = Heads(torch.load(heads_file, map_location="cpu")).to(self.device).eval()
        self.dino = AutoModel.from_pretrained("facebook/dinov2-large", dtype=torch.float16).to(self.device).eval()

    @torch.inference_mode()
    def dino_features(self, rgb):
        resized = common.resize_to_area(rgb, 448, multiple=14, max_side_factor=4)
        pixels = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0).permute(2, 0, 1)
        pixels = (pixels - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        hidden = self.dino(pixel_values=pixels[None].to(self.device).half()).last_hidden_state
        return torch.cat([hidden[0, 0], hidden[0, 1:].mean(0)]).float()

    @torch.inference_mode()
    def classify(self, image_file, title):
        with Image.open(image_file) as image:
            rgb = common.opaque_rgb_on_white(ImageOps.exif_transpose(image))
        row = {"product_title": title, "brand": "", "product_type": "", "category": "", "subcategory": ""}
        prepared = common.prepare_image(image_file, 512)
        text = prompt_text(self.processor, row, prepared, bool(title.strip()))
        batch = self.processor(text=[text], images=[prepared], return_tensors="pt").to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            apertus = apertus_features(self.apertus, self.captured, dict(batch))[0].float()
        pixel = torch.from_numpy(measure((str(image_file), False))).to(self.device)
        x = torch.cat([apertus, self.dino_features(rgb), pixel])[None]
        logits = self.heads(x)
        return {d: torch.softmax(logits[d][0].float(), 0).cpu().numpy() for d in common.DIMENSIONS}


class Board:
    """The photos of the current product and their results, shared by every page that is open."""

    def __init__(self):
        self.items, self.lock, self.version = [], threading.Lock(), 0
        self.directory = Path(tempfile.mkdtemp(prefix="galaxus_demo_"))

    def add(self, data, title):
        item_id = uuid.uuid4().hex[:10]
        path = self.directory / f"{item_id}.jpg"
        with Image.open(io.BytesIO(data)) as image:
            image = ImageOps.exif_transpose(image)
            image.thumbnail((1600, 1600))
            common.opaque_rgb_on_white(image).save(path, quality=92)
        with self.lock:
            self.items.append({"id": item_id, "path": path, "title": title, "status": "queued", "added": time.time()})
            self.version += 1
        return item_id

    def update(self, item_id, **fields):
        with self.lock:
            for item in self.items:
                if item["id"] == item_id:
                    item.update(fields)
            self.version += 1

    def reset(self):
        with self.lock:
            self.items, self.version = [], self.version + 1

    def state(self):
        with self.lock:
            items = [{k: v for k, v in item.items() if k != "path"} for item in self.items]
            return {"version": self.version, "items": items, "gallery": gallery_order(items)}


def gallery_order(items):
    """Image ids in shop order: subject first, then white backdrop, a front view, no overlay, no person."""
    done = [i for i in items if i.get("status") == "done"]

    def key(item):
        top = item["top"]
        return (SUBJECT_ORDER.index(top["subject"]), top["background"] != "white",
                PERSPECTIVE_ORDER.index(top["perspective"]), top["text_classification"] != "none",
                top["individuals"] != "none", -item["confidence"])
    ordered = sorted(done, key=key)
    return {"shown": [i["id"] for i in ordered if i["top"]["subject"] not in HIDDEN],
            "hidden": [i["id"] for i in ordered if i["top"]["subject"] in HIDDEN]}


def worker(model, board, jobs):
    while True:
        item_id = jobs.get()
        item = next((i for i in board.items if i["id"] == item_id), None)
        if item is None:
            continue
        board.update(item_id, status="running")
        try:
            started = time.time()
            probabilities = model.classify(item["path"], item["title"])
            labels = {d: [[common.CLASSES[d][k], float(p[k])] for k in np.argsort(p)[::-1][:3]]
                      for d, p in probabilities.items()}
            top = {d: v[0][0] for d, v in labels.items()}
            board.update(item_id, status="done", labels=labels, top=top, seconds=round(time.time() - started, 2),
                         confidence=float(np.mean([v[0][1] for v in labels.values()])))
        except Exception as error:  # the board shows the failure, the demo goes on
            board.update(item_id, status="error", error=str(error)[:200])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, help="a step_* folder of lora_heads.py (adapter + heads)")
    parser.add_argument("--heads", type=Path, default=Path("runs/lora_heads/init_heads.pt"), help="heads without adapter")
    parser.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="none")
    parser.add_argument("--model", default="swiss-ai/Apertus-v1.5-8B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    arguments = parser.parse_args()

    model = Model(arguments)
    board, jobs = Board(), queue.Queue()
    threading.Thread(target=worker, args=(model, board, jobs), daemon=True).start()
    app = FastAPI()

    @app.get("/")
    def page():
        return FileResponse(HERE / "index.html")

    @app.get("/logo.svg")
    def logo():
        return FileResponse(HERE / "hackapertus-logo-white.svg", media_type="image/svg+xml")

    @app.get("/partner-logo")
    def partner_logo():
        # An optional partner logo for live presentations, kept out of the repository (see .gitignore).
        path = HERE / "partner-logo.png"
        return FileResponse(path, media_type="image/png") if path.exists() else Response(status_code=404)

    @app.post("/upload")
    async def upload(files: list[UploadFile] = File(...), title: str = Form("")):
        ids = []
        for file in files:
            item_id = board.add(await file.read(), title)
            jobs.put(item_id)
            ids.append(item_id)
        return JSONResponse({"ids": ids})

    @app.get("/state")
    def state():
        return JSONResponse(board.state())

    @app.get("/image/{item_id}")
    def image(item_id: str):
        item = next((i for i in board.items if i["id"] == item_id), None)
        if item is None:
            return Response(status_code=404)
        with Image.open(item["path"]) as picture:
            picture.thumbnail((480, 480))
            buffer = io.BytesIO()
            picture.save(buffer, format="JPEG", quality=85)
        return Response(buffer.getvalue(), media_type="image/jpeg")

    @app.post("/reset")
    def reset():
        board.reset()
        return JSONResponse({"ok": True})

    print(f"demo on http://{arguments.host}:{arguments.port}", flush=True)
    uvicorn.run(app, host=arguments.host, port=arguments.port, log_level="warning")


if __name__ == "__main__":
    main()
