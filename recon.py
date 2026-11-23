"""Multi-view 3D face reconstruction from uncalibrated stills (e.g. video frames).

Pipeline:
  1. MediaPipe FaceLandmarker -> 468 dense landmarks per image (2D + a learned depth guess).
  2. Generalized Procrustes over the per-image 3D guesses -> initial shared shape S0.
  3. Bundle adjustment: one rigid shape S shared by all views, one scaled-orthographic
     camera per view, minimizing 2D reprojection error of the landmarks (visibility
     weighted). Views from different angles pin down depth geometrically instead of
     relying on the single-image depth prior; a weak prior towards S0 fixes the gauge.
  4. Vertex colors sampled from the views that see each vertex best.

Usage:
  uv run recon.py IMAGES_DIR [-o out]
  uv run recon.py --selftest
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

HERE = Path(__file__).parent
MODEL = HERE / "models" / "face_landmarker.task"
CANONICAL = HERE / "models" / "canonical_face_model.obj"
N_VERTS = 468
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------------------------------------------- mesh helpers

def load_canonical():
    verts, faces = [], []
    for line in CANONICAL.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            verts.append([float(p) for p in parts[1:4]])
        elif parts[0] == "f":
            faces.append([int(p.split("/")[0]) - 1 for p in parts[1:4]])
    verts, faces = np.array(verts), np.array(faces)
    # orient faces so normals point out of the face (towards +z of the canonical model)
    if vertex_normals(verts, faces)[:, 2].mean() < 0:
        faces = faces[:, ::-1]
    return verts, faces


def vertex_normals(verts, faces):
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)


def normalize_shape(S):
    S = S - S.mean(0)
    return S / np.sqrt((S ** 2).sum(1).mean())


def umeyama(src, dst, w=None):
    """Weighted similarity transform with dst ~= s * R @ src + t."""
    w = np.ones(len(src)) if w is None else w
    w = w / w.sum()
    mu_s, mu_d = w @ src, w @ dst
    a, b = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd((b * w[:, None]).T @ a)
    E = np.eye(3)
    E[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ E @ Vt
    s = (D * np.diag(E)).sum() / (w @ (a ** 2).sum(1))
    return s, R, mu_d - s * R @ mu_s


def visibility(S_cam, faces, floor=0.03):
    """Per-vertex weight from how much the surface faces the camera (+z = towards camera)."""
    nz = vertex_normals(S_cam, faces)[:, 2]
    return np.maximum(np.clip(nz, 0, 1) ** 0.5, floor)


# ---------------------------------------------------------------- landmarks

def detect(paths, max_faces=4):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=max_faces,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
    )
    views = []
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        for p in paths:
            bgr = cv2.imread(str(p))
            if bgr is None:
                print(f"  skip {p.name}: unreadable")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.face_landmarks:
                print(f"  skip {p.name}: no face found")
                continue
            # several people in frame: take the largest face
            cands = [np.array([[l.x * w, l.y * h, l.z * w] for l in f[:N_VERTS]]) for f in res.face_landmarks]
            pts = max(cands, key=lambda c: np.ptp(c[:, 0]) * np.ptp(c[:, 1]))
            if len(cands) > 1:
                print(f"  {p.name}: {len(cands)} faces, using the largest")
            # image coords (x right, y down, z away) -> right-handed, +z towards camera
            q = pts * np.array([1, -1, -1])
            views.append({"name": p.name, "image": rgb, "q": q, "p": q[:, :2].copy()})
    return views


# ---------------------------------------------------------------- reconstruction

def initial_shape(qs, faces, iters=8):
    ws = [visibility(q, faces) for q in qs]
    S = normalize_shape(qs[0])
    for _ in range(iters):
        acc, wsum = np.zeros_like(S), np.zeros(len(S))
        for q, w in zip(qs, ws):
            s, R, t = umeyama(q, S, w)
            acc += w[:, None] * ((s * (R @ q.T)).T + t)
            wsum += w
        S = normalize_shape(acc / wsum[:, None])
    return S


def bundle_adjust(ps, qs, faces, prior=0.15):
    """ps: per-view 2D landmarks, qs: per-view 3D guesses (same frame). Returns shape + cameras."""
    n, V = len(ps), N_VERTS
    # normalize each view so every image counts equally regardless of resolution / face size
    norms = []
    P = np.empty((n, V, 2))
    Q = []
    for i, (p, q) in enumerate(zip(ps, qs)):
        c = p.mean(0)
        sc = np.sqrt(((p - c) ** 2).sum(1).mean())
        norms.append((c, sc))
        P[i] = (p - c) / sc
        Q.append((q - np.r_[c, q[:, 2].mean()]) / sc)

    S0 = initial_shape(Q, faces)
    cams = np.zeros((n, 6))
    W = np.empty((n, V))
    for i, q in enumerate(Q):
        s, R, t = umeyama(S0, q, visibility(q, faces))
        cams[i] = [*Rotation.from_matrix(R).as_rotvec(), np.log(s), *t[:2]]
        W[i] = visibility(q, faces)

    def unpack(x):
        return x[: 6 * n].reshape(n, 6), x[6 * n:].reshape(V, 3)

    def project(cams, S):
        R = Rotation.from_rotvec(cams[:, :3]).as_matrix()
        return np.exp(cams[:, 3])[:, None, None] * np.einsum("nij,vj->nvi", R[:, :2], S) + cams[:, None, 4:6]

    def residuals(x):
        c, S = unpack(x)
        r = (project(c, S) - P) * W[..., None]
        return np.concatenate([r.ravel(), prior * (S - S0).ravel()])

    sparsity = lil_matrix((n * V * 2 + V * 3, 6 * n + V * 3), dtype=int)
    for i in range(n):
        rows = slice(i * V * 2, (i + 1) * V * 2)
        sparsity[rows, 6 * i: 6 * i + 6] = 1
        for v in range(V):
            r0 = i * V * 2 + v * 2
            sparsity[r0: r0 + 2, 6 * n + 3 * v: 6 * n + 3 * v + 3] = 1
    for v in range(V):
        for k in range(3):
            sparsity[n * V * 2 + 3 * v + k, 6 * n + 3 * v + k] = 1

    x0 = np.concatenate([cams.ravel(), S0.ravel()])
    sol = least_squares(residuals, x0, jac_sparsity=sparsity, method="trf", x_scale="jac", max_nfev=200)
    cams, S = unpack(sol.x)

    # express everything in the canonical face frame (frontal = yaw 0, upright)
    canon = normalize_shape(load_canonical()[0])
    _, Rc, _ = umeyama(S, canon)
    S, S0 = S @ Rc.T, S0 @ Rc.T
    cams[:, :3] = (Rotation.from_rotvec(cams[:, :3]) * Rotation.from_matrix(Rc.T)).as_rotvec()

    # refresh visibility with the solved poses, then report reprojection error in pixels
    R = Rotation.from_rotvec(cams[:, :3]).as_matrix()
    proj = project(cams, S)
    stats = []
    for i in range(n):
        W[i] = visibility((R[i] @ S.T).T, faces)
        c, sc = norms[i]
        err = np.linalg.norm(proj[i] - P[i], axis=1) * sc
        yaw, pitch, roll = Rotation.from_matrix(R[i]).as_euler("yxz", degrees=True)
        stats.append({
            "yaw": round(float(yaw), 1), "pitch": round(float(pitch), 1), "roll": round(float(roll), 1),
            "reproj_px": round(float((err * W[i]).sum() / W[i].sum()), 2),
        })
    proj_px = np.stack([proj[i] * norms[i][1] + norms[i][0] for i in range(n)])
    return {"S": S, "S0": S0, "R": R, "W": W, "proj_px": proj_px, "stats": stats, "cost": float(sol.cost)}


# ---------------------------------------------------------------- outputs

def vertex_colors(views, proj_px, W):
    acc, wsum = np.zeros((N_VERTS, 3)), np.zeros(N_VERTS)
    for v, pp, w in zip(views, proj_px, W):
        img = v["image"]
        h, wd = img.shape[:2]
        xs = np.clip(pp[:, 0].round().astype(int), 0, wd - 1)
        ys = np.clip((-pp[:, 1]).round().astype(int), 0, h - 1)
        wt = w ** 4  # strongly prefer the view that sees a vertex head-on
        acc += wt[:, None] * img[ys, xs]
        wsum += wt
    return (acc / wsum[:, None]).clip(0, 255).astype(np.uint8)


def write_ply(path, S, faces, colors):
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(S)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        for p, c in zip(S, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
        for t in faces:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")


def render_preview(path, S, faces, colors, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(15, 5))
    tri = S[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    base = colors[faces].mean(1) / 255 if colors is not None else np.full((len(faces), 3), 0.8)
    for k, (azim, label) in enumerate([(0, "front"), (35, "3/4"), (90, "profile")]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        # matplotlib: x right, y depth, z up -> map our (x, y up, z towards camera)
        verts = tri[..., [0, 2, 1]] * np.array([1, -1, 1])
        a = np.deg2rad(azim)
        light = np.array([np.sin(a), -np.cos(a), 0.4])
        light /= np.linalg.norm(light)
        shade = np.clip((fn[:, [0, 2, 1]] * np.array([1, -1, 1])) @ light, 0.15, 1)[:, None]
        ax.add_collection3d(Poly3DCollection(verts, facecolors=base * shade, edgecolor="none"))
        lim = np.abs(S).max()
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=5, azim=-90 + azim)
        ax.set_axis_off()
        ax.set_title(label)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_overlays(out_dir, views, proj_px, W):
    import cv2
    out_dir.mkdir(exist_ok=True)
    for v, pp, w in zip(views, proj_px, W):
        img = cv2.cvtColor(v["image"], cv2.COLOR_RGB2BGR)
        r = max(1, round(np.ptp(v["p"][:, 0]) / 150))
        for (x, y), (px, py), wi in zip(v["p"], pp, w):
            cv2.circle(img, (int(x), int(-y)), r, (0, 255, 0), -1)  # detected
            if wi > 0.2:
                cv2.circle(img, (int(px), int(-py)), r, (0, 0, 255), -1)  # reprojected 3D model
        cv2.imwrite(str(out_dir / f"{Path(v['name']).stem}.jpg"), img)


# ---------------------------------------------------------------- entry points

def run(images_dir, out_dir, prior):
    _, faces = load_canonical()
    paths = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    print(f"{len(paths)} images")
    views = detect(paths)
    if len(views) < 2:
        raise SystemExit(f"need at least 2 images with a detected face, got {len(views)}")
    print(f"face found in {len(views)} images, solving...")
    res = bundle_adjust([v["p"] for v in views], [v["q"] for v in views], faces, prior)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    colors = vertex_colors(views, res["proj_px"], res["W"])
    write_ply(out / "face.ply", res["S"], faces, colors)
    write_ply(out / "face_prior_only.ply", res["S0"], faces, colors)
    render_preview(out / "preview.png", res["S"], faces, colors, "multi-view reconstruction")
    render_preview(out / "preview_prior_only.png", res["S0"], faces, colors, "single-view depth prior (averaged)")
    write_overlays(out / "overlays", views, res["proj_px"], res["W"])
    report = {"views": [{"name": v["name"], **s} for v, s in zip(views, res["stats"])]}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    np.save(out / "shape.npy", res["S"])

    for r in report["views"]:
        print(f"  {r['name']:30s} yaw {r['yaw']:6.1f}°  pitch {r['pitch']:6.1f}°  reproj {r['reproj_px']:.2f}px")
    yaws = [r["yaw"] for r in report["views"]]
    print(f"yaw coverage {min(yaws):.0f}° .. {max(yaws):.0f}°  -> {out}/")


def selftest():
    """Synthetic check: views of a known deformed face; prior gets the *undeformed* depth.
    Multi-view BA should recover the true shape better than the prior does."""
    rng = np.random.default_rng(0)
    canon, faces = load_canonical()
    truth = normalize_shape(canon)
    # identity-like deformation: longer nose, wider jaw, deeper eye sockets
    d = truth.copy()
    d[:, 2] += 0.25 * np.exp(-((d[:, 0] / 0.25) ** 2 + ((d[:, 1] + 0.05) / 0.4) ** 2)) * np.sign(d[:, 2] + 1)
    d[:, 0] *= 1 + 0.15 * np.clip(-d[:, 1], 0, None)
    truth = normalize_shape(d)

    ps, qs = [], []
    for yaw in [-55, -30, -10, 10, 35, 60]:
        R = Rotation.from_euler("yx", [yaw, rng.normal(0, 5)], degrees=True).as_matrix()
        cam = 300 * (R @ truth.T).T + [640, -360, 0]
        p = cam[:, :2] + rng.normal(0, 0.8, (N_VERTS, 2))
        # depth guess comes from the undeformed canonical face, like a learned prior would
        q_prior = 300 * (R @ normalize_shape(canon).T).T + [640, -360, 0]
        ps.append(p)
        qs.append(np.c_[p, q_prior[:, 2]])
    res = bundle_adjust(ps, qs, faces)

    def err(S):
        s, R, t = umeyama(S, truth)
        return np.sqrt(((s * (R @ S.T).T + t - truth) ** 2).sum(1).mean())

    e0, e1 = err(res["S0"]), err(res["S"])
    print(f"shape error (rms, unit-normalized face): prior {e0:.4f} -> multi-view {e1:.4f}")
    print("reproj px per view:", [s["reproj_px"] for s in res["stats"]])
    print("recovered yaws:", [s["yaw"] for s in res["stats"]])
    assert e1 < 0.6 * e0, "bundle adjustment did not improve on the prior"
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="?", help="directory with images of one person from several angles")
    ap.add_argument("-o", "--out", default="out")
    ap.add_argument("--prior", type=float, default=0.15, help="pull towards single-view depth prior")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.images:
        run(a.images, a.out, a.prior)
    else:
        ap.error("pass an images directory or --selftest")
