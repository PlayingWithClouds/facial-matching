"""Six views of the head from MV-Adapter's image-and-geometry mode (ig2mv), with a local SDXL
checkpoint (one .safetensors file, e.g. CyberRealistic) instead of a diffusers model download.

Inputs come from avatar/head_mv.py prepare: DIR/head.obj (the fitted head) and DIR/reference.png
(the face, cut out). Writes DIR/view_<name>.png per view (front, left, back, right, top, bottom; 768 px,
grey background), DIR/views.png (all six) and DIR/geometry_pos.png / geometry_nor.png (what the views
were conditioned on, to check they line up).

Run it from MV-Adapter's folder, in its environment (ComfyUI stopped: the GPU is needed in full):

  python ../mv_head.py DIR --ckpt ~/comfy/ComfyUI/models/checkpoints/cyberrealistic_xl_play_v10.safetensors
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))  # MV-Adapter's folder
from diffusers import AutoencoderKL  # noqa: E402

from mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0  # noqa: E402
from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline  # noqa: E402
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler  # noqa: E402
from mvadapter.utils import make_image_grid  # noqa: E402
from scripts.inference_ig2mv_sdxl import run_pipeline  # noqa: E402

VIEWS = ["front", "left", "back", "right", "top", "bottom"]


def pipeline(ckpt, offload):
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    if ckpt.endswith(".safetensors"):
        pipe = MVAdapterI2MVSDXLPipeline.from_single_file(ckpt, vae=vae, torch_dtype=torch.float16)
    else:  # a diffusers model on Hugging Face, e.g. SDXL itself, which the adapter was trained on
        pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(ckpt, vae=vae, torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = ShiftSNRScheduler.from_scheduler(pipe.scheduler, shift_mode="interpolated", shift_scale=8.0)
    pipe.init_custom_adapter(num_views=len(VIEWS), self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0)
    pipe.load_custom_adapter("huanngzh/mv-adapter", weight_name="mvadapter_ig2mv_sdxl.safetensors")
    pipe.unet.to(dtype=torch.float16)  # the adapter's layers in it are loaded in float32
    pipe.to("cuda")
    if offload:
        # the text encoders on the GPU only while the prompt is encoded (~1.7 GB less). Not diffusers'
        # own enable_model_cpu_offload: its hook on the UNet copies the arguments on the way in, and the
        # dict the reference pass is meant to fill (cache_hidden_states) stays empty
        encoders = (pipe.text_encoder, pipe.text_encoder_2)
        for m in encoders:
            m.to("cpu")
        type(pipe)._execution_device = property(lambda self: torch.device("cuda"))
        encode = pipe.encode_prompt

        def encode_on_gpu(*args, **kwargs):
            for m in encoders:
                m.to("cuda")
            out = encode(*args, **kwargs)
            for m in encoders:
                m.to("cpu")
            torch.cuda.empty_cache()
            return out
        pipe.encode_prompt = encode_on_gpu
    pipe.cond_encoder.to(device="cuda", dtype=torch.float16)
    pipe.vae.enable_slicing()  # (newer diffusers dropped the pipeline's shortcut for it)
    if offload:
        split_guidance(pipe.unet)
    return pipe


def halves(x, B):
    """x (a tensor, or dicts / lists of them) cut into two along a batch of size B; anything else is
    passed to both halves as it is."""
    if torch.is_tensor(x) and x.dim() > 0 and x.shape[0] == B:
        return x.chunk(2)
    if isinstance(x, dict):
        parts = {k: halves(v, B) for k, v in x.items()}
        return tuple({k: p[i] for k, p in parts.items()} for i in range(2))
    if isinstance(x, (list, tuple)):
        parts = [halves(v, B) for v in x]
        return tuple(type(x)(p[i] for p in parts) for i in range(2))
    return x, x


def split_guidance(unet):
    """The denoising steps run the UNet on the views twice over (without and with the prompt, for
    classifier-free guidance): here as two passes of one half each, which halves the memory the
    activations and the attention across views need. Each half holds whole sets of views, so the
    attention across views sees the same as before. Single-image passes (the reference) go through
    untouched: they fill a dict that must stay the caller's."""
    forward = unet.forward

    def split_forward(sample, *args, **kwargs):
        B = sample.shape[0]
        if B < 2 or B % 2:
            return forward(sample, *args, **kwargs)
        a, b = halves((sample, args, kwargs), B)
        out = [forward(s, *ar, **kw) for s, ar, kw in (a, b)]
        if kwargs.get("return_dict", True):
            out[0].sample = torch.cat([o.sample for o in out])
            return out[0]
        return (torch.cat([o[0] for o in out]),)
    unet.forward = split_forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="from head_mv.py prepare: head.obj, reference.png")
    ap.add_argument("--ckpt", required=True, help="SDXL checkpoint: one .safetensors file, or a Hugging Face model "
                                                  "(stabilityai/stable-diffusion-xl-base-1.0, ~7 GB download)")
    ap.add_argument("--text", default="photo of a young woman's head, natural skin, soft even studio lighting, "
                                      "sharp focus")
    ap.add_argument("--negative", default="watermark, ugly, deformed, noisy, blurry, low contrast, cartoon, 3d render")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--ref-scale", type=float, default=1.0, help="how strongly the views follow the reference photo")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--offload", action="store_true", help="if the GPU runs out of memory")
    a = ap.parse_args()
    d = Path(a.dir)
    pipe = pipeline(a.ckpt, a.offload)
    images, pos, nor, ref = run_pipeline(
        pipe, mesh_path=str(d / "head.obj"), num_views=len(VIEWS), text=a.text, image=str(d / "reference.png"),
        height=768, width=768, num_inference_steps=a.steps, guidance_scale=a.cfg, seed=a.seed,
        reference_conditioning_scale=a.ref_scale, negative_prompt=a.negative)
    for name, img in zip(VIEWS, images):
        img.save(d / f"view_{name}.png")
    make_image_grid(images, rows=1).save(d / "views.png")
    make_image_grid(pos, rows=1).save(d / "geometry_pos.png")
    make_image_grid(nor, rows=1).save(d / "geometry_nor.png")
    ref.save(d / "reference_used.png")
    print(f"{d}: views.png, view_<{'|'.join(VIEWS)}>.png")


if __name__ == "__main__":
    main()
