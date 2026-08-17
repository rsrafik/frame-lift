#!/usr/bin/env python3
"""
Temporal stabilizer for upscaled image sequences.

Real-ESRGAN treats every frame independently, so it invents slightly different
fine texture each time. Across a sequence that reads as shimmer or flicker in
areas that should be perfectly still.

This pass removes it without softening motion:

  1. Motion is measured on the ORIGINAL frames, which have no hallucinated
     detail and are therefore a clean reference for what actually moved.
  2. Pixels that are static across the temporal window get replaced by the
     median of the upscaled frames in that window, which cancels the
     frame-to-frame variation while keeping the detail.
  3. Pixels that are moving are left exactly as the upscaler produced them.

Nothing is ever displaced: every operation is per-pixel, so geometry is
untouched.

    python stabilize.py Frames_4x --reference Frames -o Frames_4x_stable
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageFilter
    from tqdm import tqdm
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc.name}\n    pip install pillow numpy tqdm")

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def natural_key(path: Path) -> list:
    """Sort frame_2 before frame_10 regardless of zero padding."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def list_frames(folder: Path) -> list[Path]:
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES),
        key=natural_key,
    )


def match_references(frames: list[Path], refs: list[Path]) -> list[Path]:
    """Pair each upscaled frame with its original, by stem then by position."""
    if len(frames) != len(refs):
        raise SystemExit(
            f"error: {len(frames)} frames but {len(refs)} reference images; "
            "the two folders must correspond one to one"
        )
    by_stem = {r.stem: r for r in refs}
    paired: list[Path] = []
    for frame in frames:
        stem = frame.stem
        # Tolerate the suffix upscale.py appends by default.
        for candidate in (stem, stem.removesuffix("_upscaled")):
            if candidate in by_stem:
                paired.append(by_stem[candidate])
                break
        else:
            paired.append(refs[len(paired)])
    return paired


class FrameCache:
    """Loads frames on demand and keeps only the current window in memory."""

    def __init__(self, paths: list[Path], blur: float = 0.0) -> None:
        self.paths = paths
        self.blur = blur
        self._cache: dict[int, np.ndarray] = {}

    def get(self, index: int) -> np.ndarray:
        if index not in self._cache:
            with Image.open(self.paths[index]) as handle:
                image = handle.convert("RGB")
                # Blurring the motion reference suppresses JPEG noise, which
                # would otherwise read as motion and block stabilization.
                if self.blur > 0:
                    image = image.filter(ImageFilter.GaussianBlur(self.blur))
                self._cache[index] = np.asarray(image, dtype=np.float32)
        return self._cache[index]

    def evict_outside(self, lo: int, hi: int) -> None:
        for key in [k for k in self._cache if k < lo or k > hi]:
            del self._cache[key]


def box_blur(mask: np.ndarray, radius: int) -> np.ndarray:
    """Cheap separable box blur, used to feather the motion mask."""
    if radius < 1:
        return mask
    size = radius * 2 + 1
    padded = np.pad(mask, radius, mode="edge")
    cumulative = np.cumsum(padded, axis=0)
    blurred = (cumulative[size:, :] - cumulative[:-size, :]) / size
    padded = np.pad(blurred, ((0, 0), (radius, radius)), mode="edge")
    cumulative = np.cumsum(padded, axis=1)
    return (cumulative[:, size:] - cumulative[:, :-size]) / size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stabilize.py",
        description="Remove temporal flicker from an upscaled frame sequence.",
        epilog="example:\n"
        "  python stabilize.py Frames_4x --reference Frames -o Frames_4x_stable\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("frames", help="folder of upscaled frames, in sequence order")
    parser.add_argument(
        "--reference", required=True, metavar="DIR",
        help="folder of the ORIGINAL frames, used to detect real motion",
    )
    parser.add_argument("-o", "--output", default="stabilized", metavar="DIR", help="output folder")
    parser.add_argument(
        "--window", type=int, default=3, metavar="N",
        help="temporal window in frames, odd number (default: 3)",
    )
    parser.add_argument(
        "--threshold", type=float, default=3.0, metavar="V",
        help="reference change (0-255) still treated as fully static (default: 3.0)",
    )
    parser.add_argument(
        "--fade", type=float, default=2.5, metavar="F",
        help="stabilization fades out by threshold*F of motion (default: 2.5)",
    )
    parser.add_argument(
        "--ref-blur", type=float, default=1.0, metavar="R",
        help="blur applied to the motion reference to ignore compression noise (default: 1.0)",
    )
    parser.add_argument(
        "--strength", type=float, default=1.0, metavar="F",
        help="how much stabilization to apply in static areas, 0-1 (default: 1.0)",
    )
    parser.add_argument(
        "--feather", type=int, default=2, metavar="PX",
        help="blur radius on the motion mask, in reference pixels (default: 2)",
    )
    parser.add_argument("--quality", type=int, default=95, metavar="N", help="JPEG/WebP quality")
    args = parser.parse_args(argv)

    if args.window < 1 or args.window % 2 == 0:
        print("error: --window must be an odd number >= 1", file=sys.stderr)
        return 2
    if not 0.0 <= args.strength <= 1.0:
        print("error: --strength must be between 0 and 1", file=sys.stderr)
        return 2

    frames_dir, ref_dir = Path(args.frames).expanduser(), Path(args.reference).expanduser()
    for folder in (frames_dir, ref_dir):
        if not folder.is_dir():
            print(f"error: not a directory: {folder}", file=sys.stderr)
            return 2

    frames = list_frames(frames_dir)
    refs = match_references(frames, list_frames(ref_dir))
    if not frames:
        print("error: no images found", file=sys.stderr)
        return 1

    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.resolve() == frames_dir.resolve():
        print("error: output folder must differ from the input folder", file=sys.stderr)
        return 2

    half = args.window // 2
    hi_cache = FrameCache(frames)
    ref_cache = FrameCache(refs, blur=max(0.0, args.ref_blur))
    total_static = []

    print(f"frames    : {len(frames)}")
    print(
        f"window    : {args.window}  threshold: {args.threshold}  "
        f"fade: {args.fade}  strength: {args.strength}"
    )
    print(f"output    : {out_dir}/")

    for i in tqdm(range(len(frames)), unit="frame", desc="Stabilizing", dynamic_ncols=True):
        lo, hi = max(0, i - half), min(len(frames) - 1, i + half)
        hi_cache.evict_outside(lo, hi)
        ref_cache.evict_outside(lo, hi)

        current = hi_cache.get(i)
        neighbors = list(range(lo, hi + 1))

        if len(neighbors) == 1 or args.strength == 0:
            result = current
            total_static.append(0.0)
        else:
            ref_current = ref_cache.get(i)
            # Motion = the largest change seen anywhere in the window, so a
            # pixel only counts as static if it is still across every frame.
            motion = np.zeros(ref_current.shape[:2], dtype=np.float32)
            for j in neighbors:
                if j == i:
                    continue
                delta = np.abs(ref_cache.get(j) - ref_current).mean(axis=2)
                motion = np.maximum(motion, delta)

            # Full strength up to `threshold`, then a linear fade to zero so
            # the transition into moving areas is not visible as a hard edge.
            lo = max(args.threshold, 1e-6)
            hi_t = max(lo * max(args.fade, 1.0 + 1e-6), lo + 1e-6)
            static = np.clip((hi_t - motion) / (hi_t - lo), 0.0, 1.0)
            static = box_blur(static, args.feather)
            total_static.append(float(static.mean()))

            # Median across the window cancels the per-frame hallucination
            # without the softening a mean would introduce.
            stacked = np.stack([hi_cache.get(j) for j in neighbors], axis=0)
            temporal = np.median(stacked, axis=0)

            weight = static * args.strength
            if weight.shape != current.shape[:2]:
                mask_img = Image.fromarray((weight * 255.0).astype(np.uint8), mode="L")
                mask_img = mask_img.resize((current.shape[1], current.shape[0]), RESAMPLE_BILINEAR)
                weight = np.asarray(mask_img, dtype=np.float32) / 255.0
            weight = weight[:, :, None]

            result = current * (1.0 - weight) + temporal * weight

        destination = out_dir / frames[i].name
        image = Image.fromarray(np.clip(result, 0, 255).round().astype(np.uint8), mode="RGB")
        suffix = destination.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            image.save(destination, quality=args.quality, subsampling=0, optimize=True)
        elif suffix == ".webp":
            image.save(destination, quality=args.quality, method=6)
        else:
            image.save(destination)

    print(
        f"\ndone: {len(frames)} frames, "
        f"average stabilization weight {100 * np.mean(total_static):.0f}%"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
