"""Render an episode's MP4s into a README GIF: agent view with a wrist-view inset.

    python scripts/make_gif.py EPISODE_DIR OUT.gif [--size 320] [--stride 4]

EPISODE_DIR holds agentview.mp4 and wrist.mp4 (frames as the policy sees them).
Requires imageio + imageio-ffmpeg + Pillow (local tooling only, not the runtime).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image


def frames(path, stride):
    with imageio.get_reader(str(path)) as reader:
        return [frame for i, frame in enumerate(reader) if i % stride == 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--colors", type=int, default=64)
    args = parser.parse_args()
    agent = frames(args.episode / "agentview.mp4", args.stride)
    wrist = frames(args.episode / "wrist.mp4", args.stride)
    inset, border = args.size * 3 // 10, 2
    images = []
    for a, w in zip(agent, wrist):
        canvas = Image.fromarray(a).resize((args.size, args.size), Image.LANCZOS)
        x = y = args.size - inset - border - 4
        canvas.paste(
            (255, 255, 255), (x - border, y - border, x + inset + border, y + inset + border)
        )
        canvas.paste(Image.fromarray(w).resize((inset, inset), Image.LANCZOS), (x, y))
        images.append(canvas.quantize(colors=args.colors))
    images += [images[-1]] * 8  # hold the final frame
    args.out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        args.out, save_all=True, append_images=images[1:], duration=160, loop=0, optimize=True
    )
    print(args.out, len(images), "frames", args.out.stat().st_size // 1024, "KiB")


if __name__ == "__main__":
    main()
