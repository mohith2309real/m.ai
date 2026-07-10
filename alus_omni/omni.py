#!/usr/bin/env python3
"""
ALUS OMNI — m.ai's all-in-one model: text + image generation + video generation.

    python alus_omni/omni.py text  "explain recursion"
    python alus_omni/omni.py image "a potato supercomputer in a basement" --out potato.png
    python alus_omni/omni.py video "a rock learning to code" --seconds 30 --out rock.mp4

Text runs anywhere on the m.ai registry weights — our own models only. Image and video need a GPU + one-time weight
downloads (pip install diffusers transformers accelerate safetensors, and
ffmpeg on PATH for video). On CPU they will technically run and practically
teach you patience.

Video strategy (honest engineering): text-to-video models that make one long
coherent shot don't fit in a basement budget. Alus generates N short clips
(img2img-drifted frames per scene) and ffmpeg-concats them into a single
30-60s MP4. It's how half the "AI video" you see online is made anyway.
"""
import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMG_MODEL = os.environ.get("ALUS_IMAGE_MODEL", "stabilityai/sd-turbo")
FPS = 8
FRAMES_PER_SCENE = 24  # 3s per scene at 8fps


# ── text ───────────────────────────────────────────────────────────────
def do_text(prompt, args):
    # our own weights only — no external labs
    os.execv(sys.executable, [sys.executable, os.path.join(REPO, "mai_run.py"),
                              args.model, prompt])


# ── image ──────────────────────────────────────────────────────────────
def _pipe(device):
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except ImportError:
        raise SystemExit("Image/video gen needs: pip install diffusers transformers "
                         "accelerate safetensors (and a GPU unless you enjoy waiting).")
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(IMG_MODEL, torch_dtype=dtype)
    return pipe.to(device)


def do_image(prompt, args):
    pipe = _pipe(args.device)
    img = pipe(prompt=prompt, num_inference_steps=args.steps, guidance_scale=0.0).images[0]
    img.save(args.out)
    print(f"[alus] image saved -> {args.out}")


# ── video ──────────────────────────────────────────────────────────────
def do_video(prompt, args):
    import shutil
    import tempfile
    if not shutil.which("ffmpeg"):
        raise SystemExit("Video gen needs ffmpeg on PATH.")
    seconds = max(5, min(args.seconds, 60))
    n_scenes = max(1, round(seconds / (FRAMES_PER_SCENE / FPS)))
    pipe = _pipe(args.device)

    workdir = tempfile.mkdtemp(prefix="alus_video_")
    frame_i = 0
    print(f"[alus] {seconds}s video = {n_scenes} scenes x {FRAMES_PER_SCENE} frames @ {FPS}fps")
    for s in range(n_scenes):
        scene_prompt = f"{prompt}, cinematic, scene {s + 1} of {n_scenes}"
        base = pipe(prompt=scene_prompt, num_inference_steps=args.steps,
                    guidance_scale=0.0).images[0]
        for f in range(FRAMES_PER_SCENE):
            # slow zoom for motion — crop a shrinking center window and resize back
            z = 1.0 + 0.15 * (f / FRAMES_PER_SCENE)
            w, h = base.size
            cw, ch = int(w / z), int(h / z)
            x, y = (w - cw) // 2, (h - ch) // 2
            base.crop((x, y, x + cw, y + ch)).resize((w, h)).save(
                os.path.join(workdir, f"frame_{frame_i:05d}.png"))
            frame_i += 1
        print(f"[alus] scene {s + 1}/{n_scenes} done")

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(workdir, "frame_%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", args.out], check=True)
    shutil.rmtree(workdir)
    print(f"[alus] video saved -> {args.out} ({frame_i / FPS:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["text", "image", "video"])
    ap.add_argument("prompt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seconds", type=int, default=30, help="video length (5-60)")
    ap.add_argument("--steps", type=int, default=2, help="diffusion steps (sd-turbo: 1-4)")
    ap.add_argument("--device", default=None, help="cuda or cpu (auto)")
    ap.add_argument("--model", default="alus-1.0", help="registry model for text mode")
    args = ap.parse_args()

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"
    if args.out is None:
        args.out = {"image": "alus_image.png", "video": "alus_video.mp4"}.get(args.mode)

    {"text": do_text, "image": do_image, "video": do_video}[args.mode](args.prompt, args)


if __name__ == "__main__":
    main()
