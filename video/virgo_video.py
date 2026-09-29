"""Virgo 1.0 video: text to video with real motion (Wan 2.1, 1.3B, 480p, about 5 seconds).

    python video/virgo_video.py "a horse running on the beach at sunset, cinematic" horse.mp4

Needs an NVIDIA GPU with about 8 GB of VRAM or more (with CPU offload); a clip takes a few minutes.
It does not run on a CPU in reasonable time.
"""
import argparse

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

BASE = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
NEGATIVE = "blurry, low quality, distorted, static, still image, watermark, text, subtitles"


class VirgoVideo:
    def __init__(self, base=BASE):
        if not torch.cuda.is_available():
            raise SystemExit("Virgo video needs an NVIDIA GPU (8 GB+ VRAM).")
        vae = AutoencoderKLWan.from_pretrained(base, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanPipeline.from_pretrained(base, vae=vae, torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload()  # fits smaller GPUs

    def generate(self, prompt, seconds=5, fps=16, width=832, height=480, steps=30, seed=None):
        generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None
        frames = int(seconds * fps) // 4 * 4 + 1  # Wan wants 4k + 1 frames
        return self.pipe(prompt=prompt, negative_prompt=NEGATIVE, height=height, width=width, num_frames=frames,
                         num_inference_steps=steps, guidance_scale=5.0, generator=generator).frames[0], fps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("out")
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    frames, fps = VirgoVideo().generate(args.prompt, args.seconds, seed=args.seed)
    export_to_video(frames, args.out, fps=fps)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
