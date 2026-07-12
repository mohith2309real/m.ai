#!/usr/bin/env python3
"""
ALUS 0.2 — the omni generator, Kaggle edition. Images + up to 60s of video,
as realistic as free hardware gets.

HOW TO RUN ON KAGGLE (one-time setup, 2 minutes):
  1. kaggle.com -> Create -> Notebook
  2. Settings (right panel) -> Accelerator -> GPU T4 x2   <- GPU, not TPU!
     (SDXL/SVD are CUDA models; on a TPU they simply don't run. The TPU is
      for training the next text model — different project, different day.)
  3. Settings -> Internet -> ON (needs phone verification once)
  4. First cell:
       !pip -q install diffusers transformers accelerate safetensors
       !git clone https://github.com/mohith2309/m.ai.git
  5. Second cell:
       !python m.ai/alus_omni/kaggle_omni.py video "a golden retriever \
            running on a beach at sunset, cinematic" --seconds 60
     or images:
       !python m.ai/alus_omni/kaggle_omni.py image "a potato datacenter" -n 4

  Everything lands in /kaggle/working/alus_out — download from the sidebar.

SAVE-ITERS / RESUME: every video clip is written to disk the second it
finishes. If the session dies at clip 12/18, rerun the SAME command and it
continues from clip 12. Nothing is ever lost.

HONEST ENGINEERING, v0.2 edition: one long coherent AI shot still doesn't
exist on a basement budget — so Alus generates real motion in ~3.5s bursts
(Stable Video Diffusion, the strongest open video model that fits free
hardware) and chains them: the last frame of each clip seeds the next.
Over a minute the scene slowly evolves. That's not a bug, that's the
state of the art you can own for $0.

  images: SDXL-Turbo (--hq: full SDXL, slower, sharper)
  video:  SDXL keyframe -> SVD-XT img2vid, chained -> ffmpeg concat
  60s on a T4: roughly 18 clips x 2-4 min = about an hour. Fits a session.
"""
import argparse
import json
import os
import subprocess
import sys

# SVD-XT is trained for 25-frame bursts at 1024x576, ~7fps native
WIDTH, HEIGHT = 1024, 576
FRAMES = 25
NATIVE_FPS = 7

IMG_TURBO = "stabilityai/sdxl-turbo"
IMG_HQ = "stabilityai/stable-diffusion-xl-base-1.0"
VID_MODEL = "stabilityai/stable-video-diffusion-img2vid-xt"


def device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"  # local mac: works, teaches patience
    return "cpu"      # works in the way a rowboat crosses an ocean


def image_pipe(hq):
    import torch
    from diffusers import AutoPipelineForText2Image
    dev = device()
    pipe = AutoPipelineForText2Image.from_pretrained(
        IMG_HQ if hq else IMG_TURBO,
        torch_dtype=torch.float16 if dev != "cpu" else torch.float32,
        variant="fp16" if dev != "cpu" else None,
    )
    if dev == "cuda":
        pipe.enable_model_cpu_offload()  # fits a 16GB T4 with room to spare
    else:
        pipe.to(dev)
    return pipe


def gen_keyframe(prompt, out_png, seed, hq):
    import torch
    pipe = image_pipe(hq)
    g = torch.Generator().manual_seed(seed)
    kwargs = dict(prompt=prompt, width=WIDTH, height=HEIGHT, generator=g)
    if hq:
        kwargs.update(num_inference_steps=30, guidance_scale=7.0)
    else:
        kwargs.update(num_inference_steps=4, guidance_scale=0.0)
    img = pipe(**kwargs).images[0]
    img.save(out_png)
    del pipe
    free_vram()
    return img


def free_vram():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def do_image(args):
    os.makedirs(args.out, exist_ok=True)
    for i in range(args.n):
        path = os.path.join(args.out, f"image_{i:02d}.png")
        if os.path.exists(path):
            print(f"[alus] {path} exists — skipping (resume)")
            continue
        print(f"[alus] image {i + 1}/{args.n}…")
        gen_keyframe(args.prompt, path, args.seed + i, args.hq)
        print(f"[alus]   saved {path}")
    print(f"[alus] done -> {args.out}")


def do_video(args):
    import torch
    from diffusers import StableVideoDiffusionPipeline
    from diffusers.utils import export_to_video, load_image

    out = args.out
    clips_dir = os.path.join(out, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    manifest_path = os.path.join(out, "manifest.json")

    # one clip contributes FRAMES-1 new frames (its first frame = the seed)
    n_clips = max(1, -(-args.seconds * NATIVE_FPS // (FRAMES - 1)))
    print(f"[alus] {args.seconds}s @ {NATIVE_FPS}fps native -> {n_clips} clips")

    # keyframe: generate once, reuse across resumes
    key_png = os.path.join(out, "keyframe.png")
    if not os.path.exists(key_png):
        print(f"[alus] keyframe: {args.prompt!r}")
        gen_keyframe(args.prompt, key_png, args.seed, hq=True)  # keyframe deserves HQ
        print(f"[alus]   saved {key_png}")

    dev = device()
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        VID_MODEL,
        torch_dtype=torch.float16 if dev != "cpu" else torch.float32,
        variant="fp16" if dev != "cpu" else None,
    )
    if dev == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(dev)

    cond_png = key_png
    for c in range(n_clips):
        clip_mp4 = os.path.join(clips_dir, f"clip_{c:03d}.mp4")
        last_png = os.path.join(clips_dir, f"last_{c:03d}.png")
        if os.path.exists(clip_mp4) and os.path.exists(last_png):
            print(f"[alus] clip {c + 1}/{n_clips} exists — skipping (resume)")
            cond_png = last_png
            continue
        print(f"[alus] clip {c + 1}/{n_clips}…")
        cond = load_image(cond_png).resize((WIDTH, HEIGHT))
        g = torch.Generator().manual_seed(args.seed + c)
        frames = pipe(
            cond, decode_chunk_size=2, generator=g,
            motion_bucket_id=args.motion, noise_aug_strength=0.02,
            num_frames=FRAMES,
        ).frames[0]
        frames[-1].save(last_png)          # next clip's seed — saved FIRST so
        export_to_video(frames, clip_mp4, fps=NATIVE_FPS)  # a crash mid-write resumes clean
        cond_png = last_png
        with open(manifest_path, "w") as f:
            json.dump({"prompt": args.prompt, "seed": args.seed,
                       "clips_done": c + 1, "clips_total": n_clips}, f)
        print(f"[alus]   saved {clip_mp4}")

    del pipe
    free_vram()
    concat(clips_dir, n_clips, out, args)


def concat(clips_dir, n_clips, out, args):
    lst = os.path.join(out, "concat.txt")
    with open(lst, "w") as f:
        for c in range(n_clips):
            f.write(f"file 'clips/clip_{c:03d}.mp4'\n")
    raw = os.path.join(out, "alus_raw.mp4")
    final = os.path.join(out, "alus_final.mp4")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", raw])
    if args.smooth:
        # motion-interpolate 7fps -> 24fps: big smoothness win, ~10-20 min of CPU
        print("[alus] smoothing to 24fps (this is the slow, worth-it part)…")
        run(["ffmpeg", "-y", "-i", raw, "-vf",
             "minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:vsbmc=1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", final])
    else:
        run(["ffmpeg", "-y", "-i", raw, "-c:v", "libx264", "-pix_fmt",
             "yuv420p", "-r", "24", final])
    print(f"[alus] FINAL -> {final}")


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"[alus] command failed: {' '.join(cmd[:4])}…\n{r.stderr[-800:]}")


def main():
    ap = argparse.ArgumentParser(description="Alus 0.2 omni generator (Kaggle)")
    ap.add_argument("mode", choices=["image", "video"])
    ap.add_argument("prompt")
    ap.add_argument("--out", default=None, help="output dir")
    ap.add_argument("--seconds", type=int, default=60, help="video length")
    ap.add_argument("-n", type=int, default=1, help="image count")
    ap.add_argument("--hq", action="store_true", help="images: full SDXL (slower, sharper)")
    ap.add_argument("--motion", type=int, default=127,
                    help="SVD motion_bucket_id: 1=still life, 255=action cam")
    ap.add_argument("--smooth", action="store_true", default=True,
                    help="motion-interpolate the final cut to 24fps")
    ap.add_argument("--no-smooth", dest="smooth", action="store_false")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if args.out is None:
        base = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
        args.out = os.path.join(base, "alus_out")
    if device() == "cpu":
        print("[alus] WARNING: no GPU found. On Kaggle: Settings -> "
              "Accelerator -> GPU T4 x2. Proceeding anyway, glacially.")

    (do_image if args.mode == "image" else do_video)(args)


if __name__ == "__main__":
    main()
