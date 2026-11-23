"""Pull face crops of one person out of a video, ready for recon.py.

Samples a frame every --every seconds, detects faces (YuNet), embeds them (SFace) and
groups them by identity. The target is either the person matching --ref, or by default
the identity with the most screen time. From the target's faces it keeps sharp frames
spread over head angles.

Usage:
  uv run extract.py VIDEO [-o images] [--every 0.5] [--ref face.jpg] [--cluster K] [--max 30]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
YUNET = HERE / "models" / "face_detection_yunet_2023mar.onnx"
SFACE = HERE / "models" / "face_recognition_sface_2021dec.onnx"
SAME_PERSON = 0.363  # SFace cosine threshold from the OpenCV model card
LINK_LOOSE = 0.2  # weaker match accepted if the face is also where the target just was


def make_models():
    det = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), score_threshold=0.7)
    rec = cv2.FaceRecognizerSF.create(str(SFACE), "")
    return det, rec


def detect(det, rec, img, min_size):
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    out = []
    for f in faces if faces is not None else []:
        x, y, bw, bh = f[:4]
        if min(bw, bh) < min_size:
            continue
        emb = rec.feature(rec.alignCrop(img, f)).ravel()
        out.append({"box": f[:4].copy(), "lm": f[4:14].reshape(5, 2).copy(), "emb": emb / np.linalg.norm(emb)})
    return out


def rough_yaw(lm):
    """-1 (looking to image left) .. +1 (image right) from where the nose sits between the eyes."""
    reye, leye, nose = lm[0], lm[1], lm[2]
    span = leye[0] - reye[0]
    if abs(span) < 1e-6:
        return 0.0
    return float(np.clip(2 * (nose[0] - reye[0]) / span - 1, -1, 1))


def sharpness(img, box):
    x, y, w, h = box.astype(int)
    crop = cv2.cvtColor(img[max(y, 0): y + h, max(x, 0): x + w], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else 0.0


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def cluster(faces):
    """Greedy identity clustering on embedding centroids."""
    centroids, members = [], []
    for i, f in enumerate(faces):
        sims = [c @ f["emb"] / np.linalg.norm(c) for c in centroids]
        k = int(np.argmax(sims)) if sims else -1
        if k >= 0 and sims[k] > SAME_PERSON:
            centroids[k] += f["emb"]
            members[k].append(i)
        else:
            centroids.append(f["emb"].copy())
            members.append([i])
    order = sorted(range(len(members)), key=lambda k: -len(members[k]))
    return [members[k] for k in order], [centroids[k] / np.linalg.norm(centroids[k]) for k in order]


def crop_face(img, box, pad=1.2):
    """Square crop around the face with generous margin so landmarking has context."""
    x, y, w, h = box
    cx, cy, s = x + w / 2, y + h / 2, max(w, h) * (1 + pad)
    x0, y0 = int(max(cx - s / 2, 0)), int(max(cy - s / 2, 0))
    return img[y0: int(min(cy + s / 2, img.shape[0])), x0: int(min(cx + s / 2, img.shape[1]))]


def contact_sheet(path, thumbs, cols=8, size=96):
    tiles = [cv2.resize(t, (size, size)) for t in thumbs[: cols * 2]]
    while len(tiles) % cols:
        tiles.append(np.zeros((size, size, 3), np.uint8))
    rows = [np.hstack(tiles[i: i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(path), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-o", "--out", default="images")
    ap.add_argument("--every", type=float, default=0.5, help="seconds between sampled frames")
    ap.add_argument("--ref", help="image of the target face; otherwise the most frequent face is used")
    ap.add_argument("--cluster", type=int, help="pick this identity cluster (see clusters/ sheets)")
    ap.add_argument("--max", type=int, default=30, help="max frames to keep")
    ap.add_argument("--min-size", type=int, default=80, help="ignore faces smaller than this (px)")
    a = ap.parse_args()

    det, rec = make_models()
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, round(fps * a.every))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{total} frames @ {fps:.1f} fps, sampling every {step} frames")

    frames, faces = {}, []
    for idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if not ok:
            break
        found = detect(det, rec, img, a.min_size)
        for f in found:
            f["frame"] = idx
            f["sharp"] = sharpness(img, f["box"])
            f["yaw"] = rough_yaw(f["lm"])
            faces.append(f)
        if found:
            frames[idx] = img
    cap.release()
    print(f"{len(faces)} faces in {len(frames)} sampled frames")
    if not faces:
        raise SystemExit("no faces found")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    members, centroids = cluster(faces)
    sheets = out.parent / f"{out.name}_clusters"
    sheets.mkdir(exist_ok=True)
    for k, m in enumerate(members[:6]):
        contact_sheet(sheets / f"cluster_{k}.jpg", [crop_face(frames[faces[i]["frame"]], faces[i]["box"], 0.3) for i in m])
        print(f"  cluster {k}: {len(m)} faces")

    if a.ref:
        ref = detect(det, rec, cv2.imread(a.ref), 20)
        if not ref:
            raise SystemExit("no face in --ref image")
        target = max(ref, key=lambda f: f["box"][2] * f["box"][3])["emb"]
    else:
        target = centroids[a.cluster or 0]

    # accept confident matches, plus weaker ones sitting where the target was in the previous sample
    picked, last = [], None
    for f in sorted(faces, key=lambda f: f["frame"]):
        sim = float(f["emb"] @ target)
        near = last is not None and f["frame"] - last["frame"] <= step and iou(f["box"], last["box"]) > 0.3
        if sim > SAME_PERSON or (near and sim > LINK_LOOSE):
            f["sim"] = sim
            picked.append(f)
            last = f
    # one face per frame (the best match), drop blurry ones
    best = {}
    for f in picked:
        if f["frame"] not in best or f["sim"] > best[f["frame"]]["sim"]:
            best[f["frame"]] = f
    picked = list(best.values())
    if picked:
        cut = np.percentile([f["sharp"] for f in picked], 30)
        picked = [f for f in picked if f["sharp"] >= cut]
    print(f"target matched in {len(best)} frames, {len(picked)} sharp enough")

    # spread the selection over head angles: bin by rough yaw, take sharpest per bin round-robin
    bins = {}
    for f in sorted(picked, key=lambda f: -f["sharp"]):
        bins.setdefault(int((f["yaw"] + 1) * 4.99), []).append(f)
    chosen = []
    while len(chosen) < a.max and any(bins.values()):
        for b in sorted(bins):
            if bins[b] and len(chosen) < a.max:
                chosen.append(bins[b].pop(0))

    for old in out.glob("f*.jpg"):
        old.unlink()
    meta = []
    for f in sorted(chosen, key=lambda f: f["frame"]):
        name = f"f{f['frame']:07d}.jpg"
        cv2.imwrite(str(out / name), crop_face(frames[f["frame"]], f["box"]))
        meta.append({"file": name, "frame": f["frame"], "t": round(f["frame"] / fps, 2),
                     "sim": round(f["sim"], 3), "yaw_rough": round(f["yaw"], 2), "sharp": round(f["sharp"], 1)})
    (out / "frames.json").write_text(json.dumps(meta, indent=2))
    yaws = [m["yaw_rough"] for m in meta]
    if meta:
        print(f"wrote {len(meta)} crops to {out}/ (rough yaw {min(yaws):+.2f} .. {max(yaws):+.2f})")
    print(f"identity contact sheets in {sheets}/ — rerun with --cluster K if the wrong person was picked")


if __name__ == "__main__":
    main()
