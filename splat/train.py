"""Train a Gaussian splat of a head from a prepared capture (see prepare.py).

Background is masked out: the loss only looks at the person, and an alpha loss pushes
everything outside the mask to transparent, so the result is the head alone.

Usage:
  uv run train.py CAPTURE_DIR [--steps 15000] [--sh 3]
Outputs CAPTURE_DIR/splat.ply (standard 3DGS format) and CAPTURE_DIR/turntable.mp4
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from tqdm import tqdm

dev = torch.device("cuda")


# ---------------------------------------------------------------- data

def load_capture(root, max_side):
    root = Path(root)
    rec = pycolmap.Reconstruction(str(root / "sparse"))
    views = []
    for img in sorted(rec.images.values(), key=lambda i: i.name):
        cam = rec.cameras[img.camera_id]
        rgb = cv2.cvtColor(cv2.imread(str(root / "images" / img.name)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(root / "masks" / (Path(img.name).stem + ".png")), cv2.IMREAD_GRAYSCALE)
        K = cam.calibration_matrix().copy()
        s = min(1.0, max_side / max(rgb.shape[:2]))
        if s < 1:
            rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            K[:2] *= s
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
        w2c = np.eye(4)
        w2c[:3] = img.cam_from_world().matrix()
        views.append({
            "name": img.name,
            "rgb": torch.from_numpy(rgb).to(dev),
            "mask": torch.from_numpy(mask).to(dev),
            "K": torch.tensor(K, dtype=torch.float32, device=dev),
            "w2c": torch.tensor(w2c, dtype=torch.float32, device=dev),
            "w2c_np": w2c,
            "K_np": K,
            "hw": rgb.shape[:2],
            "points2D": [(p.xy * s, p.point3D_id) for p in img.points2D if p.has_point3D()],
        })
    return rec, views


def head_points(rec, views, min_frac=0.6):
    """SfM points that land inside the person mask in most views that observe them."""
    votes = {}
    for v in views:
        m, (h, w) = v["mask"].cpu().numpy(), v["hw"]
        for xy, pid in v["points2D"]:
            x, y = int(xy[0]), int(xy[1])
            if 0 <= x < w and 0 <= y < h:
                inside, total = votes.get(pid, (0, 0))
                votes[pid] = (inside + (m[y, x] > 127), total + 1)
    keep = [pid for pid, (i, t) in votes.items() if t >= 2 and i / t >= min_frac]
    xyz = np.array([rec.points3D[p].xyz for p in keep])
    rgb = np.array([rec.points3D[p].color for p in keep]) / 255.0
    return xyz, rgb


# ---------------------------------------------------------------- model

C0 = 0.28209479177387814  # SH band 0


def init_params(xyz, rgb, sh_degree, n_random, scene_scale):
    # pad sparse skin coverage with random points inside the (slightly grown) head bbox
    lo, hi = np.percentile(xyz, 2, 0), np.percentile(xyz, 98, 0)
    pad = (hi - lo) * 0.15
    rnd = np.random.uniform(lo - pad, hi + pad, (n_random, 3))
    xyz = np.concatenate([xyz, rnd])
    rgb = np.concatenate([rgb, np.full((n_random, 3), 0.5)])

    means = torch.tensor(xyz, dtype=torch.float32, device=dev)
    d = torch.cat([torch.cdist(c, means).topk(4, largest=False).values[:, 1:] for c in means.split(4096)])
    scales = torch.log(d.mean(1).clamp_min(1e-7))[:, None].repeat(1, 3)
    n = len(means)
    sh0 = ((torch.tensor(rgb, dtype=torch.float32, device=dev) - 0.5) / C0)[:, None]
    shN = torch.zeros(n, (sh_degree + 1) ** 2 - 1, 3, device=dev)
    params = torch.nn.ParameterDict({
        "means": means, "scales": scales,
        "quats": F.normalize(torch.randn(n, 4, device=dev), dim=-1),
        "opacities": torch.logit(torch.full((n,), 0.1, device=dev)),
        "sh0": sh0, "shN": shN,
    })
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
           "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15) for k, lr in lrs.items()}
    return params, opts


def render(params, w2c, K, hw, sh_degree):
    colors = torch.cat([params["sh0"], params["shN"]], 1)
    out, alpha, info = rasterization(
        params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
        torch.sigmoid(params["opacities"]), colors, w2c[None], K[None], hw[1], hw[0],
        sh_degree=sh_degree,  # no background -> composited on black
    )
    return out[0], alpha[0, ..., 0], info


def ssim(a, b, win=11):
    """a, b: [H, W, 3] in 0..1"""
    a, b = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    g = torch.exp(-((torch.arange(win, device=dev) - win // 2) ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    w = (g[:, None] @ g[None])[None, None].repeat(3, 1, 1, 1)
    f = lambda x: F.conv2d(x, w, padding=win // 2, groups=3)
    mu_a, mu_b = f(a), f(b)
    va, vb, cov = f(a * a) - mu_a ** 2, f(b * b) - mu_b ** 2, f(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))).mean()


# ---------------------------------------------------------------- export

def save_ply(path, params):
    means = params["means"].detach().cpu().numpy()
    sh0 = params["sh0"].detach().cpu().numpy().reshape(len(means), -1)
    shN = params["shN"].detach().transpose(1, 2).cpu().numpy().reshape(len(means), -1)
    op = params["opacities"].detach().cpu().numpy()[:, None]
    sc = params["scales"].detach().cpu().numpy()
    q = F.normalize(params["quats"].detach(), dim=-1).cpu().numpy()
    cols = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)] + \
           [f"f_rest_{i}" for i in range(shN.shape[1])] + ["opacity"] + \
           [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    data = np.concatenate([means, np.zeros_like(means), sh0, shN, op, sc, q], 1).astype(np.float32)
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {len(data)}\n".encode())
        for c in cols:
            f.write(f"property float {c}\n".encode())
        f.write(b"end_header\n")
        f.write(data.tobytes())


def turntable(path, params, views, center, sh_degree, frames=120):
    """Orbit around the head at the capture distance, starting from the average viewing direction."""
    c2ws = [np.linalg.inv(v["w2c_np"]) for v in views]
    eyes = np.array([m[:3, 3] for m in c2ws])
    up = -np.mean([m[:3, 1] for m in c2ws], 0)
    up /= np.linalg.norm(up)
    front = (eyes - center).mean(0)
    front -= up * (front @ up)
    front /= np.linalg.norm(front)
    side = np.cross(up, front)
    radius = np.median(np.linalg.norm(eyes - center, axis=1))
    v0 = views[0]
    K, (h, w) = v0["K"], v0["hw"]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    with torch.no_grad():
        for i in range(frames):
            a = 2 * math.pi * i / frames
            eye = center + radius * (math.cos(a) * front + math.sin(a) * side)
            z = center - eye
            z /= np.linalg.norm(z)
            y = -up - (-up @ z) * z
            y /= np.linalg.norm(y)
            x = np.cross(y, z)
            w2c = np.eye(4)
            w2c[:3, :3] = np.stack([x, y, z])
            w2c[:3, 3] = -w2c[:3, :3] @ eye
            img, _, _ = render(params, torch.tensor(w2c, dtype=torch.float32, device=dev), K, (h, w), sh_degree)
            out.write(cv2.cvtColor((img.clamp(0, 1) * 255).byte().cpu().numpy(), cv2.COLOR_RGB2BGR))
    out.release()


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--sh", type=int, default=3, help="spherical harmonics degree")
    ap.add_argument("--max-side", type=int, default=1600, help="downscale training images")
    ap.add_argument("--random-points", type=int, default=20000)
    a = ap.parse_args()

    root = Path(a.capture)
    rec, views = load_capture(root, a.max_side)
    xyz, rgb = head_points(rec, views)
    print(f"{len(views)} posed views, {len(xyz)} SfM points on the head")
    if len(xyz) < 50:
        raise SystemExit("too few SfM points inside the person mask — capture probably failed to register")
    center = np.median(xyz, 0)
    eyes = np.array([np.linalg.inv(v["w2c_np"])[:3, 3] for v in views])
    scene_scale = float(np.linalg.norm(eyes - center, axis=1).max()) * 1.1

    params, opts = init_params(xyz, rgb, a.sh, a.random_points, scene_scale)
    strategy = DefaultStrategy(refine_stop_iter=int(a.steps * 0.6), reset_every=3000, verbose=False)
    strategy.check_sanity(params, opts)
    state = strategy.initialize_state(scene_scale=scene_scale)
    sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1 / a.steps))

    bar = tqdm(range(a.steps))
    for step in bar:
        v = views[np.random.randint(len(views))]
        sh_deg = min(a.sh, step // 1000)  # grow view-dependent color gradually
        img, alpha, info = render(params, v["w2c"], v["K"], v["hw"], sh_deg)
        mask = v["mask"].float()[..., None] / 255
        gt = v["rgb"].float() / 255 * mask  # person on black
        strategy.step_pre_backward(params, opts, state, step, info)
        loss = 0.8 * (img - gt).abs().mean() + 0.2 * (1 - ssim(img, gt)) + 0.5 * (alpha - mask[..., 0]).abs().mean()
        loss.backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        sched.step()
        strategy.step_post_backward(params, opts, state, step, info, packed=True)  # rasterization packs by default
        if step % 100 == 0:
            bar.set_postfix(loss=f"{loss.item():.4f}", n=len(params["means"]))

    # drop near-invisible gaussians before export
    with torch.no_grad():
        keep = torch.sigmoid(params["opacities"]) > 0.02
        params = torch.nn.ParameterDict({k: p[keep] for k, p in params.items()})
    save_ply(root / "splat.ply", params)
    print(f"saved {keep.sum().item()} gaussians -> {root / 'splat.ply'}")

    with torch.no_grad():
        psnrs = []
        for v in views[:: max(1, len(views) // 10)]:
            img, _, _ = render(params, v["w2c"], v["K"], v["hw"], a.sh)
            gt = v["rgb"].float() / 255 * (v["mask"].float()[..., None] / 255)
            psnrs.append(-10 * torch.log10(((img - gt) ** 2).mean()).item())
    print(f"train-view PSNR {np.mean(psnrs):.2f} dB")
    turntable(root / "turntable.mp4", params, views, center, a.sh)
    print(f"turntable -> {root / 'turntable.mp4'}")


if __name__ == "__main__":
    main()
