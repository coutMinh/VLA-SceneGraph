"""Render an episode's agent-view + wrist MP4s into a small captioned GIF for the README.

    python scripts/make_gif.py EPISODE_DIR OUT.gif --caption "S2-t01-s00 · failure"

Requires imageio + imageio-ffmpeg + Pillow (local tooling only, not the runtime).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def frames(path, stride):
    with imageio.get_reader(str(path)) as reader:
        return [frame for i, frame in enumerate(reader) if i % stride == 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--caption", default="")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--size", type=int, default=160)
    args = parser.parse_args()
    agent = frames(args.episode / "agentview.mp4", args.stride)
    wrist = frames(args.episode / "wrist.mp4", args.stride)
    images = []
    for a, w in zip(agent, wrist):
        pair = Image.fromarray(np.concatenate([a, w], 1)).resize((2 * args.size, args.size))
        canvas = Image.new("RGB", (2 * args.size, args.size + 18), "white")
        canvas.paste(pair, (0, 18))
        ImageDraw.Draw(canvas).text((4, 3), args.caption, fill="black")
        images.append(canvas.quantize(colors=64))
    images += [images[-1]] * 8  # hold the final frame
    args.out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        args.out, save_all=True, append_images=images[1:], duration=150, loop=0, optimize=True
    )
    print(args.out, len(images), "frames", args.out.stat().st_size // 1024, "KiB")


if __name__ == "__main__":
    main()
