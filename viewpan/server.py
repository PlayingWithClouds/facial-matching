"""Turn one photo to new viewpoints: its 3D point map -> turned and splatted (holes, stretching) -> Difix
(nvidia/difix_ref, the photo as reference) repairs it.

  ../MoGe/.venv/bin/python depth_server.py            (MoGe-2 on :8011; optional, see below)
  ../Difix3D/.venv/bin/python server.py [--port 8010]

then open http://localhost:8010. Upload an image, drag on the views to turn. The batch section runs Difix over a
folder of avatar renders (RGBAvatar render_vhap.py --dump), each with its real frame as the reference. POST /fix
repairs one image (RGBAvatar anny_viewer.py's live Difix preview).

The point map comes from MoGe-2 (depth_server.py: metric 3D points and the camera, sharp on faces). Without it,
from Depth Anything (relative disparity d in [0, 1], 1 nearest): z = 1 - 0.3 d through a 50 degree camera. The
page's depth factor stretches the depths about the pivot (the median depth of the nearest third of the points),
which the view turns about.
"""

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "Difix3D" / "src"))
MAX_SIDE = 768  # (the long side the photo is worked at; a multiple of 8, as the VAE needs)

app = FastAPI()
gpu = threading.Lock()
models = {}
state = {}


def load_models():
    from pipeline_difix import DifixPipeline
    from transformers import pipeline
    models["difix"] = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True,
                                                    torch_dtype=torch.float16).to("cuda")  # (fp16: ~2.6 GB, not 5.2)
    models["depth"] = pipeline("depth-estimation", model="LiheYoung/depth-anything-small-hf", device=0)


def data_url(img, fmt="JPEG"):
    buf = io.BytesIO()
    img.save(buf, fmt, quality=92)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def rotation(yaw, pitch):
    y, p = np.radians(yaw), np.radians(pitch)
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return Rx @ Ry


def moge_points(img, url):
    """MoGe-2's point map of `img` from depth_server.py: points [H, W, 3], mask [H, W], K [3, 3] in pixels."""
    import requests
    buf = io.BytesIO()
    img.save(buf, "PNG")
    r = requests.post(url, files={"file": ("img.png", buf.getvalue())}, timeout=60)
    r.raise_for_status()
    d = np.load(io.BytesIO(r.content))
    K = d["intrinsics"] * [[img.width], [img.height], [1]]
    return d["points"], d["mask"] & np.isfinite(d["points"]).all(-1), K


def depth_anything_points(img):
    """The fallback: Depth Anything's relative disparity as a point map through a 50 degree camera."""
    with gpu:
        pred = models["depth"](img)["predicted_depth"]
    d = torch.nn.functional.interpolate(pred[None], size=img.size[::-1], mode="bicubic")[0, 0].cpu().numpy()
    d = (d - d.min()) / max(d.max() - d.min(), 1e-6)
    h, w = d.shape
    f = 0.5 * max(h, w) / np.tan(np.radians(25))
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    z = 1 - 0.3 * d
    v, u = np.mgrid[0:h, 0:w] + 0.5
    return np.stack([(u - w / 2) * z / f, (v - h / 2) * z / f, z], -1), np.ones_like(d, bool), K


def warp(yaw, pitch, depth_scale):
    """The photo seen from the camera turned by yaw/pitch (degrees) about the pivot, its depths stretched by
    depth_scale about the pivot's: RGB uint8 [H, W, 3] with the holes black, and the hole mask."""
    rgb, pts, valid, K = state["rgb"], state["points"], state["valid"], state["K"]
    h, w = valid.shape
    P, col = pts[valid], rgb[valid]
    near = P[:, 2] <= np.quantile(P[:, 2], 1 / 3)
    pivot = np.median(P[near], 0)
    P = P * ((pivot[2] + (P[:, 2] - pivot[2]) * depth_scale) / P[:, 2])[:, None]  # (along each pixel's ray)
    Q = (P - pivot) @ rotation(yaw, pitch).T + pivot  # (turning the scene the other way: the camera turns)
    ok = Q[:, 2] > 1e-3
    x = Q[ok, 0] / Q[ok, 2] * K[0, 0] + K[0, 2] - 0.5
    y = Q[ok, 1] / Q[ok, 2] * K[1, 1] + K[1, 2] - 0.5
    col, depth = col[ok], Q[ok, 2]
    order = np.argsort(-depth)  # (far first: the nearest point is written last and wins)
    out = np.zeros((h, w, 3), np.uint8)
    hit = np.zeros((h, w), bool)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):  # (each point covers 2x2 pixels: fewer cracks)
        xi, yi = np.floor(x[order]).astype(int) + dx, np.floor(y[order]).astype(int) + dy
        inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        out[yi[inside], xi[inside]] = col[order][inside]
        hit[yi[inside], xi[inside]] = True
    return out, ~hit


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    s = MAX_SIDE / max(img.size)
    size = (max(8, round(img.width * s / 8) * 8), max(8, round(img.height * s / 8) * 8))
    img = img.resize(size, Image.LANCZOS)
    try:
        pts, valid, K = moge_points(img, state["moge_url"])
        source = "MoGe-2"
    except Exception as e:  # (depth_server.py not running)
        print("MoGe failed, falling back to Depth Anything:", e)
        pts, valid, K = depth_anything_points(img)
        source = "Depth Anything (fallback)"
    state.update(img=img, rgb=np.asarray(img), points=pts, valid=valid, K=K)
    z = np.where(valid, pts[..., 2], np.nan)
    lo, hi = np.nanquantile(z, [0.01, 0.99])
    shade = np.nan_to_num(1 - (z - lo) / max(hi - lo, 1e-6), nan=0).clip(0, 1)  # (near: white)
    depth_img = Image.fromarray((shade * 255).astype(np.uint8))
    fov = np.degrees(2 * np.arctan(0.5 * img.width / K[0, 0]))  # (horizontal)
    return {"width": img.width, "height": img.height, "image": data_url(img), "depth": data_url(depth_img, "PNG"),
            "source": source, "fov": round(float(fov), 1)}


@app.get("/view")
def view(yaw: float = 0, pitch: float = 0, depth: float = 1.0, fix: int = 0):
    if "img" not in state:
        return JSONResponse({"error": "upload an image first"}, 400)
    t0 = time.time()
    out, holes = warp(yaw, pitch, depth)
    warped = Image.fromarray(out)
    res = {"warp": data_url(warped), "holes": round(float(holes.mean()), 3), "warp_ms": round(1000 * (time.time() - t0))}
    if fix:
        t1 = time.time()
        with gpu:
            fixed = models["difix"]("remove degradation", image=warped, ref_image=state["img"], num_inference_steps=1,
                                    timesteps=[199], guidance_scale=0.0).images[0]
        res.update(fixed=data_url(fixed.resize(warped.size, Image.LANCZOS)), fix_ms=round(1000 * (time.time() - t1)))
    return res


class Fix(BaseModel):
    image: str  # (data URLs)
    ref: str | None = None


def from_data_url(s):
    return Image.open(io.BytesIO(base64.b64decode(s.split(",", 1)[1]))).convert("RGB")


@app.post("/fix")
def fix(req: Fix):
    """Difix on one image (e.g. an avatar viewer's render), with a reference (its nearest real frame) or without
    (then the image is its own); answers the repaired image at the image's size."""
    t0 = time.time()
    img = from_data_url(req.image)
    ref = from_data_url(req.ref) if req.ref else img
    x = multiple_of_8(img)
    with gpu:
        out = difix_batch([x], [ref.resize(x.size, Image.LANCZOS)])[0]
    return {"fixed": data_url(out.resize(img.size, Image.LANCZOS)), "ms": round(1000 * (time.time() - t0))}


@torch.no_grad()
def difix_batch(images, refs, ref_keys=None):
    """Difix on several (image, reference) pairs in one pass; the same as the pipeline one pair at a time.

    The pipeline itself batches wrongly with references: it stacks [images, references], but its UNet's
    multi-view attention pairs neighbours ((b v) -> b, v=2), so the latents are interleaved here: img0, ref0,
    img1, ref1, ... Only the images are decoded, with their own encoder activations (the VAE's decoder adds
    the last encoding's: saved right after encoding the images). A reference with a key (ref_keys: e.g. its
    file) is encoded once and kept, as several views share one."""
    from pipeline_difix import retrieve_latents, retrieve_timesteps
    pipe = models["difix"]
    dev, dtype, sf = pipe.device, pipe.unet.dtype, pipe.vae.config.scaling_factor
    proc = pipe.image_processor
    cache = models.setdefault("ref_latents", {})
    if len(cache) > 256:
        cache.clear()
    lat_r = []
    for key, ref in zip(ref_keys or [None] * len(refs), refs):
        if key is None or key not in cache:
            z = retrieve_latents(pipe.vae.encode(proc.preprocess(ref).to(dev, dtype))) * sf
            if key is None:
                lat_r.append(z)
                continue
            cache[key] = z
        lat_r.append(cache[key])
    lat_i = retrieve_latents(pipe.vae.encode(proc.preprocess(images).to(dev, dtype))) * sf
    skips = pipe.vae.encoder.current_down_blocks  # (the images' own, for the decoder)
    lat = torch.stack([lat_i, torch.cat(lat_r)], 1).flatten(0, 1)  # (img0, ref0, img1, ref1, ...)
    if "prompt" not in models:
        models["prompt"] = pipe.encode_prompt("remove degradation", dev, 1, False)[0]
    timesteps, _ = retrieve_timesteps(pipe.scheduler, 1, dev, [199])
    t = timesteps[0]
    noise = pipe.unet(pipe.scheduler.scale_model_input(lat, t), t,
                      encoder_hidden_states=models["prompt"].expand(len(lat), -1, -1), return_dict=False)[0]
    lat = pipe.scheduler.step(noise, t, lat, return_dict=False)[0][0::2]
    pipe.vae.decoder.incoming_skip_acts = skips
    out = pipe.vae.decode(lat / sf, return_dict=False)[0]
    return proc.postprocess(out, output_type="pil")


def multiple_of_8(img):
    w, h = img.size
    return img.resize((max(8, round(w / 8) * 8), max(8, round(h / 8) * 8)), Image.LANCZOS)


@app.get("/batch")
def batch(path: str, force: int = 0, quiet: int = 0, bs: int = 4):
    """Difix over a render batch (RGBAvatar render_vhap.py --dump: PATH/manifest.json, each render with its
    frame as the reference), bs pairs at a time (difix_batch); the results are kept in PATH/fixed/ (force=1:
    redo them; quiet=1: answer with the counts only, for batches too big to send back)."""
    root = Path(path)
    if not (root / "manifest.json").exists():
        return JSONResponse({"error": f"no manifest.json in {root}"}, 400)
    m = json.loads((root / "manifest.json").read_text())
    (root / "fixed").mkdir(exist_ok=True)
    out, t0 = [], time.time()
    dst = lambda it: root / "fixed" / (Path(it["name"]).stem + ".jpg")
    todo = [it for it in m["items"] if force or not dst(it).exists()]
    for s in range(0, len(todo), bs):
        chunk = todo[s:s + bs]
        renders = [Image.open(root / it["render"]).convert("RGB") for it in chunk]
        refs = [multiple_of_8(Image.open(root / it["ref"]).convert("RGB")).resize(multiple_of_8(r).size) for it, r in
                zip(chunk, renders)]
        with gpu:
            fixed = difix_batch([multiple_of_8(r) for r in renders], refs,
                                [f'{root / it["ref"]}@{ref.size}' for it, ref in zip(chunk, refs)])
        for it, r, f in zip(chunk, renders, fixed):
            f.resize(r.size, Image.LANCZOS).save(dst(it), quality=92)
    n = len(todo)
    if quiet:
        return {"items": len(m["items"]), "fixed_now": n, "ms": round(1000 * (time.time() - t0))}
    for it in m["items"]:
        out.append({**it, "render": data_url(Image.open(root / it["render"]).convert("RGB")),
                    "fixed": data_url(Image.open(dst(it)))})
    refs = {it["ref"]: data_url(Image.open(root / it["ref"]).convert("RGB")) for it in m["items"]}
    return {"items": out, "refs": refs, "yaws": m["yaws"], "frames": m["frames"], "work": m.get("work"),
            "fixed_now": n, "ms": round(1000 * (time.time() - t0))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--moge", default="http://127.0.0.1:8011/points", help="depth_server.py's address")
    a = ap.parse_args()
    state["moge_url"] = a.moge
    load_models()
    uvicorn.run(app, host="127.0.0.1", port=a.port)
