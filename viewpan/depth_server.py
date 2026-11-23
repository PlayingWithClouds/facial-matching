"""MoGe-2 as a service for server.py: an image in, its 3D point map out (camera coordinates, OpenCV: x right,
y down, z forward, metres), with the valid-pixel mask and the camera's intrinsics.

Runs in MoGe's environment (MoGe-2 wants newer libraries than Difix's pinned ones):

  ../MoGe/.venv/bin/python depth_server.py [--port 8011]
"""

import argparse
import io

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from PIL import Image

app = FastAPI()
model = {}


@app.post("/points")
async def points(file: UploadFile = File(...)):
    """npz: points [H, W, 3] float32, mask [H, W] bool, intrinsics [3, 3] (normalised: x by W, y by H)."""
    img = np.asarray(Image.open(io.BytesIO(await file.read())).convert("RGB"))
    x = torch.tensor(img / 255, dtype=torch.float32, device="cuda").permute(2, 0, 1)
    with torch.no_grad():
        out = model["moge"].infer(x)
    buf = io.BytesIO()
    np.savez(buf, points=out["points"].float().cpu().numpy(), mask=out["mask"].cpu().numpy().astype(bool),
             intrinsics=out["intrinsics"].float().cpu().numpy())
    return Response(buf.getvalue(), media_type="application/octet-stream")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8011)
    a = ap.parse_args()
    from moge.model.v2 import MoGeModel
    model["moge"] = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda").eval()
    uvicorn.run(app, host="127.0.0.1", port=a.port)
