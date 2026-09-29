"""Virgo 1.0 images: text to image with SD-Turbo (fast: 1–4 steps).

    python image/virgo_image.py "a cute robot reading a book, watercolor" robot.png
    python image/virgo_image.py "Angkor Wat at sunrise" out.png --steps 4 --lora image/out/virgo-style

A GPU makes an image in about a second; a CPU takes around 10–60 seconds. --lora loads an optional
Virgo style LoRA (see image/README.md). Prompts should be in English; the website already
translates Khmer prompts before sending them.
"""
import argparse

import torch
from diffusers import AutoPipelineForText2Image

BASE = "stabilityai/sd-turbo"


class VirgoImage:
    def __init__(self, base=BASE, lora=None):
        gpu = torch.cuda.is_available()
        self.pipe = AutoPipelineForText2Image.from_pretrained(base, torch_dtype=torch.float16 if gpu else torch.float32)
        self.pipe.to("cuda" if gpu else "cpu")
        if lora:
            self.pipe.load_lora_weights(lora)

    def generate(self, prompt, steps=2, size=512, seed=None):
        generator = torch.Generator(self.pipe.device).manual_seed(seed) if seed is not None else None
        # SD-Turbo is trained without classifier-free guidance: guidance_scale must stay 0.
        return self.pipe(prompt, num_inference_steps=steps, guidance_scale=0.0, width=size, height=size, generator=generator).images[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("out")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--seed", type=int)
    p.add_argument("--lora")
    args = p.parse_args()
    VirgoImage(lora=args.lora).generate(args.prompt, args.steps, args.size, args.seed).save(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
