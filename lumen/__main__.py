import argparse
import os
import sys
import time

from .loader import load_scene
from .png import write_png
from .render import render, to_srgb8


def ansi_preview(px, cols=80):
    # two pixels per character cell using the upper-half block, truecolor escapes
    h, w, _ = px.shape
    step = max(1, w // cols)
    small = px[::step, ::step]
    lines = []
    for y in range(0, small.shape[0] - 1, 2):
        row = []
        for x in range(small.shape[1]):
            t, b = small[y, x], small[y + 1, x]
            row.append(f"\x1b[38;2;{t[0]};{t[1]};{t[2]}m\x1b[48;2;{b[0]};{b[1]};{b[2]}m▀")
        lines.append("".join(row) + "\x1b[0m")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lumen", description="tiny vectorised ray tracer")
    ap.add_argument("scene", help="path to a scene .json")
    ap.add_argument("-o", "--out", default="render.png")
    ap.add_argument("-W", "--width", type=int, default=480)
    ap.add_argument("-H", "--height", type=int, default=270)
    ap.add_argument("-s", "--samples", type=int, default=4, help="anti-aliasing passes")
    ap.add_argument("-d", "--depth", type=int, default=3, help="max reflection bounces")
    ap.add_argument("--preview", action="store_true", help="also draw it in the terminal")
    args = ap.parse_args(argv)

    scene = load_scene(args.scene)
    t0 = time.perf_counter()
    img = render(scene, args.width, args.height, args.samples, args.depth)
    dt = time.perf_counter() - t0
    px = to_srgb8(img)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_png(args.out, px)
    rays = args.width * args.height * args.samples
    print(f"{args.width}x{args.height}, {args.samples} spp, depth {args.depth}: "
          f"{dt:.2f}s ({rays / dt / 1e3:.0f}k primary rays/s) -> {args.out}")
    if args.preview:
        print(ansi_preview(px))
    return 0


if __name__ == "__main__":
    sys.exit(main())
