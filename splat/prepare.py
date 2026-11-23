"""Turn a video (or a folder of photos) of a still head into a posed, masked capture for train.py.

  frames  : ffmpeg sample, keep the sharper frame of each consecutive pair
  poses   : COLMAP SfM (SIFT, sequential + loop matching), single OPENCV camera
  undistort to pinhole images
  masks   : MediaPipe selfie segmentation, head + hair + neck only (no clothes, no arms)

Usage:
  uv run prepare.py VIDEO_OR_DIR CAPTURE_DIR [--fps 3] [--no-mask]
"""

import argparse
import shutil
import subprocess
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pycolmap

HERE = Path(__file__).parent
SEG_MODEL = HERE / "models" / "selfie_multiclass_256x256.tflite"
SEG_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
           "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def sharpness(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (640, int(640 * g.shape[0] / g.shape[1])))
    return cv2.Laplacian(g, cv2.CV_64F).var()


def extract_frames(src, dst, fps, max_side):
    dst.mkdir(parents=True, exist_ok=True)
    src = Path(src)
    if src.is_dir():
        for p in sorted(src.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                img = cv2.imread(str(p))
                s = min(1.0, max_side / max(img.shape[:2]))
                cv2.imwrite(str(dst / f"{p.stem}.jpg"), cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
        return
    tmp = dst.parent / "frames_all"
    tmp.mkdir(exist_ok=True)
    scale = f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-vf", f"fps={fps * 2},{scale}",
                    "-q:v", "2", str(tmp / "%05d.jpg")], check=True)
    frames = sorted(tmp.glob("*.jpg"))
    for a, b in zip(frames[::2], frames[1::2] + [None]):
        best = a if b is None or sharpness(a) >= sharpness(b) else b
        shutil.copy(best, dst / best.name)
    shutil.rmtree(tmp)


def run_sfm(frames, work, sequential):
    db = work / "database.db"
    db.unlink(missing_ok=True)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"
    pycolmap.extract_features(db, frames, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader)
    if sequential:
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = 15
        pairing.loop_detection = False
        pycolmap.match_sequential(db, pairing_options=pairing)
    else:
        pycolmap.match_exhaustive(db)
    sparse = work / "sparse"
    shutil.rmtree(sparse, ignore_errors=True)
    sparse.mkdir()
    recs = pycolmap.incremental_mapping(db, frames, sparse)
    if not recs:
        raise SystemExit("SfM failed: no reconstruction (too little overlap, blur, or the scene moved)")
    k, rec = max(recs.items(), key=lambda kv: kv[1].num_reg_images())
    n = len(list(frames.iterdir()))
    print(f"SfM: registered {rec.num_reg_images()}/{n} frames, {rec.num_points3D()} points"
          + (f" ({len(recs)} disconnected models, using the largest)" if len(recs) > 1 else ""))
    return sparse / str(k)


def head_mask(confs, h, w):
    """confs: background, hair, body-skin, face-skin, clothes, others. Keep the head and a neck-sized
    region of body skin under the face, so arms holding the phone and torso are excluded."""
    bg, hair, body, face, clothes, other = confs
    head = np.clip(hair + face + other, 0, 1)
    face_px = np.argwhere(face > 0.5)
    if len(face_px) < 50:
        return np.zeros((h, w), np.uint8)
    (y0, x0), (y1, x1) = face_px.min(0), face_px.max(0)
    fw, fh = x1 - x0, y1 - y0
    neck = np.zeros_like(body)
    cx = (x0 + x1) // 2
    neck[y0 + fh // 2: min(h, y1 + int(0.6 * fh)), max(0, cx - int(0.65 * fw)): cx + int(0.65 * fw)] = 1
    m = np.clip(head + body * neck, 0, 1)
    # keep the connected blob that contains the face
    hard = (m > 0.5).astype(np.uint8)
    _, labels = cv2.connectedComponents(hard)
    lab = np.bincount(labels[face > 0.5].ravel()).argmax()
    m = m * (labels == lab)
    return (m * 255).astype(np.uint8)


def make_masks(images, masks):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    if not SEG_MODEL.exists():
        SEG_MODEL.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(SEG_URL, SEG_MODEL)
    masks.mkdir(exist_ok=True)
    opts = vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=str(SEG_MODEL)),
                                        output_confidence_masks=True, output_category_mask=False)
    empty = 0
    with vision.ImageSegmenter.create_from_options(opts) as seg:
        for p in sorted(images.iterdir()):
            rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            res = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            confs = [cv2.resize(c.numpy_view(), (w, h)) for c in res.confidence_masks]
            m = head_mask(confs, h, w)
            empty += m.max() == 0
            cv2.imwrite(str(masks / f"{p.stem}.png"), m)
    if empty:
        print(f"warning: no face found for the mask in {empty} frames (those frames only teach 'transparent')")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="video file or folder of photos")
    ap.add_argument("out", help="capture directory to create")
    ap.add_argument("--fps", type=float, default=3, help="frames per second of video to keep")
    ap.add_argument("--max-side", type=int, default=1920)
    ap.add_argument("--exhaustive", action="store_true", help="match all pairs (photos in no particular order)")
    ap.add_argument("--no-mask", action="store_true", help="train on the full image (scenes, testing)")
    a = ap.parse_args()

    out = Path(a.out)
    work = out / "work"
    frames = work / "frames"
    shutil.rmtree(frames, ignore_errors=True)
    extract_frames(a.src, frames, a.fps, a.max_side)
    print(f"{len(list(frames.iterdir()))} frames")

    model = run_sfm(frames, work, sequential=not (a.exhaustive or Path(a.src).is_dir()))
    for d in ("images", "sparse", "masks"):
        shutil.rmtree(out / d, ignore_errors=True)
    pycolmap.undistort_images(out, model, frames)

    if a.no_mask:
        (out / "masks").mkdir()
        for p in (out / "images").iterdir():
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(out / "masks" / f"{p.stem}.png"), np.full_like(img, 255))
    else:
        make_masks(out / "images", out / "masks")
    print(f"capture ready: {out}  ->  uv run train.py {out}")


if __name__ == "__main__":
    main()
