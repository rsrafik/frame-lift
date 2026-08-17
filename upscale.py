#!/usr/bin/env python3
"""
Batch AI image upscaler built on Real-ESRGAN.

Standalone: the network architectures are defined in this file, so the only
runtime dependencies are torch, Pillow, numpy and tqdm. Pretrained weights are
downloaded automatically on first use and cached in ~/.cache/upscale-models.

    python upscale.py ./photos -o ./photos_4x
    python upscale.py a.jpg b.png --model anime --scale 2

Run `python upscale.py --help` for the full option list.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import ssl
import sys
import tempfile
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image, ImageOps, UnidentifiedImageError
    from tqdm import tqdm
except ImportError as exc:
    sys.exit(
        f"Missing dependency: {exc.name}\n\n"
        "Install the requirements first:\n"
        "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "    pip install pillow numpy tqdm\n\n"
        "See the README for CUDA / Apple Silicon instructions."
    )

# Pillow >= 9.1 moved the resampling enums; support both spellings.
try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC
    RESAMPLE_LANCZOS = Image.LANCZOS

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
ALPHA_CAPABLE_SUFFIXES = {".png", ".webp", ".tif", ".tiff"}
CACHE_DIR = Path(os.environ.get("UPSCALE_MODEL_DIR", Path.home() / ".cache" / "upscale-models"))


# The four architecture classes below are reimplemented from the upstream
# reference code so this project does not depend on basicsr at runtime:
#   RRDBNet and its blocks -- BasicSR (Apache-2.0), basicsr/archs/rrdbnet_arch.py
#   SRVGGNetCompact        -- Real-ESRGAN (BSD-3-Clause), realesrgan/archs/srvgg_arch.py
# Layer names and shapes are kept identical so the official checkpoints load
# unchanged. See NOTICE.md for the full attribution.


class ResidualDenseBlock(nn.Module):
    """Five-conv dense block with residual scaling, as used inside an RRDB."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Generator behind RealESRGAN_x4plus / x2plus / x4plus_anime_6B."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch *= 4
        elif scale == 1:
            num_in_ch *= 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feat = F.pixel_unshuffle(x, downscale_factor=2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(x, downscale_factor=4)
        else:
            feat = x

        feat = self.conv_first(feat)
        feat = feat + self.conv_body(self.body(feat))
        # These two steps always upsample 4x; the pixel_unshuffle above trades
        # spatial resolution for channels to give the 2x and 1x variants.
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class SRVGGNetCompact(nn.Module):
    """Small VGG-style generator behind realesr-general-x4v3 (the 'fast' model)."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
    ) -> None:
        super().__init__()
        self.upscale = upscale
        body: list[nn.Module] = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_feat)]
        for _ in range(num_conv):
            body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            body.append(nn.PReLU(num_feat))
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


_RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    url: str
    scale: int
    description: str
    builder: str  # "rrdb" or "srvgg"
    num_block: int = 23
    num_feat: int = 64
    num_conv: int = 32


MODELS: dict[str, ModelSpec] = {
    "general": ModelSpec(
        name="general",
        filename="RealESRGAN_x4plus.pth",
        url=f"{_RELEASES}/v0.1.0/RealESRGAN_x4plus.pth",
        scale=4,
        description="Best all-round quality for photos and general images (4x).",
        builder="rrdb",
    ),
    "general-x2": ModelSpec(
        name="general-x2",
        filename="RealESRGAN_x2plus.pth",
        url=f"{_RELEASES}/v0.2.1/RealESRGAN_x2plus.pth",
        scale=2,
        description="Native 2x version of the general model; faster, less VRAM.",
        builder="rrdb",
    ),
    "anime": ModelSpec(
        name="anime",
        filename="RealESRGAN_x4plus_anime_6B.pth",
        url=f"{_RELEASES}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        scale=4,
        description="Tuned for anime, illustration and line art (4x, 6 blocks).",
        builder="rrdb",
        num_block=6,
    ),
    "fast": ModelSpec(
        name="fast",
        filename="realesr-general-x4v3.pth",
        url=f"{_RELEASES}/v0.2.5.0/realesr-general-x4v3.pth",
        scale=4,
        description="Lightweight 4x model; much faster on CPU, slightly softer.",
        builder="srvgg",
        num_conv=32,
    ),
}


def build_network(spec: ModelSpec) -> nn.Module:
    if spec.builder == "rrdb":
        return RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            scale=spec.scale,
            num_feat=spec.num_feat,
            num_block=spec.num_block,
        )
    return SRVGGNetCompact(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=spec.num_feat,
        num_conv=spec.num_conv,
        upscale=spec.scale,
    )


def _ssl_context() -> "ssl.SSLContext":
    """Build a verifying SSL context, preferring certifi's CA bundle.

    Framework Python builds on macOS often ship without a usable CA store,
    which makes urllib fail on any HTTPS download until the user runs
    "Install Certificates.command"; certifi sidesteps that entirely.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download_weights(spec: ModelSpec, cache_dir: Path) -> Path:
    """Fetch the checkpoint once and cache it on disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / spec.filename
    if target.exists() and target.stat().st_size > 0:
        return target

    print(f"Downloading {spec.filename} -> {target}")
    tmp_fd, tmp_name = tempfile.mkstemp(dir=cache_dir, suffix=".part")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        request = urllib.request.Request(spec.url, headers={"User-Agent": "upscale.py"})
        context = _ssl_context()
        with urllib.request.urlopen(request, context=context) as response, open(tmp_path, "wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            with tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=spec.filename,
                leave=False,
            ) as bar:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    bar.update(len(chunk))
        shutil.move(str(tmp_path), target)
    except (urllib.error.URLError, OSError) as exc:
        tmp_path.unlink(missing_ok=True)
        hint = ""
        if isinstance(getattr(exc, "reason", None), ssl.SSLCertVerificationError) or isinstance(
            exc, ssl.SSLCertVerificationError
        ):
            hint = "\nTLS verification failed. Run `pip install certifi` and try again."
        raise RuntimeError(
            f"Could not download {spec.url}: {exc}{hint}\n"
            f"Alternatively download it manually and place it at {target}"
        ) from exc
    return target


def load_state_dict(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("params_ema", "params", "state_dict"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            checkpoint = checkpoint[key]
            break
    # Strip a DataParallel prefix if the checkpoint was saved that way.
    return {k[7:] if k.startswith("module.") else k: v for k, v in checkpoint.items()}


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        index = device.index or 0
        props = torch.cuda.get_device_properties(index)
        return f"CUDA ({props.name}, {props.total_memory / 1024 ** 3:.1f} GB)"
    if device.type == "mps":
        return "Apple Silicon (MPS)"
    return f"CPU ({os.cpu_count()} cores)"


def is_oom_error(exc: BaseException) -> bool:
    if torch.cuda.is_available() and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "alloc" in message and "fail" in message


def empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


class Upscaler:
    """Runs a Real-ESRGAN network over images, with tiling and OOM fallback."""

    def __init__(
        self,
        spec: ModelSpec,
        device: torch.device,
        half: bool,
        tile: int,
        tile_pad: int,
        cache_dir: Path = CACHE_DIR,
    ) -> None:
        self.spec = spec
        self.device = device
        self.scale = spec.scale
        self.tile = tile
        self.tile_pad = tile_pad
        self.half = half

        weights = download_weights(spec, cache_dir)
        model = build_network(spec)
        model.load_state_dict(load_state_dict(weights), strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.dtype = torch.float16 if half else torch.float32
        self.model = model.to(device=device, dtype=self.dtype)

        # RRDBNet's 2x/1x variants consume pixels via pixel_unshuffle, so the
        # input to the network must be divisible by this factor.
        if spec.builder == "rrdb" and spec.scale == 2:
            self.mod_pad = 2
        elif spec.builder == "rrdb" and spec.scale == 1:
            self.mod_pad = 4
        else:
            self.mod_pad = 1

    def _forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run the network, padding the input up to the required multiple."""
        height, width = tensor.shape[2:]
        pad_h = (self.mod_pad - height % self.mod_pad) % self.mod_pad
        pad_w = (self.mod_pad - width % self.mod_pad) % self.mod_pad
        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
        output = self.model(tensor)
        if pad_h or pad_w:
            output = output[:, :, : height * self.scale, : width * self.scale]
        return output

    def _tiled_forward(self, tensor: torch.Tensor, tile: int) -> torch.Tensor:
        """Process the image tile by tile so VRAM use stays bounded."""
        batch, channels, height, width = tensor.shape
        output = tensor.new_zeros((batch, channels, height * self.scale, width * self.scale))
        tiles_x = math.ceil(width / tile)
        tiles_y = math.ceil(height / tile)

        for y in range(tiles_y):
            for x in range(tiles_x):
                start_x, start_y = x * tile, y * tile
                end_x, end_y = min(start_x + tile, width), min(start_y + tile, height)

                # Overlap each tile with its neighbors, then discard the
                # padding after inference to avoid visible seams.
                pad_start_x = max(start_x - self.tile_pad, 0)
                pad_end_x = min(end_x + self.tile_pad, width)
                pad_start_y = max(start_y - self.tile_pad, 0)
                pad_end_y = min(end_y + self.tile_pad, height)

                patch = tensor[:, :, pad_start_y:pad_end_y, pad_start_x:pad_end_x]
                result = self._forward(patch)

                crop_x = (start_x - pad_start_x) * self.scale
                crop_y = (start_y - pad_start_y) * self.scale
                output[
                    :,
                    :,
                    start_y * self.scale : end_y * self.scale,
                    start_x * self.scale : end_x * self.scale,
                ] = result[
                    :,
                    :,
                    crop_y : crop_y + (end_y - start_y) * self.scale,
                    crop_x : crop_x + (end_x - start_x) * self.scale,
                ]
        return output

    @torch.inference_mode()
    def _infer(self, array: np.ndarray) -> np.ndarray:
        """Upscale an HxWx3 float32 array in [0, 1]."""
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=self.dtype)

        if self.tile > 0:
            attempts = [self.tile]
        else:
            # Auto mode: try the whole image first, then shrink on OOM.
            attempts = [0, 512, 256, 128]

        for index, attempt in enumerate(attempts):
            is_last = index == len(attempts) - 1
            try:
                if attempt == 0:
                    output = self._forward(tensor)
                else:
                    output = self._tiled_forward(tensor, attempt)
                break
            except RuntimeError as exc:
                if is_last or not is_oom_error(exc):
                    raise
                empty_device_cache(self.device)
                tqdm.write(
                    f"  out of memory, retrying with {attempts[index + 1]}px tiles"
                )

        output = output.squeeze(0).permute(1, 2, 0).clamp_(0, 1)
        return output.to(dtype=torch.float32, device="cpu").numpy()

    def upscale_image(self, image: Image.Image, outscale: float, alpha_mode: str) -> Image.Image:
        alpha: Image.Image | None = None
        if image.mode in ("RGBA", "LA", "PA") or (
            image.mode == "P" and "transparency" in image.info
        ):
            image = image.convert("RGBA")
            alpha = image.getchannel("A")
            rgb = image.convert("RGB")
        else:
            rgb = image.convert("RGB")

        array = np.asarray(rgb, dtype=np.float32) / 255.0
        upscaled = self._infer(array)
        result = Image.fromarray((upscaled * 255.0).round().astype(np.uint8), mode="RGB")

        if alpha is not None:
            if alpha_mode == "model":
                # Feed the alpha channel through the network as a gray image so
                # the matte gets the same treatment as the color data.
                alpha_arr = np.asarray(alpha, dtype=np.float32) / 255.0
                alpha_rgb = np.repeat(alpha_arr[:, :, None], 3, axis=2)
                alpha_up = self._infer(alpha_rgb)[:, :, 0]
                alpha_img = Image.fromarray((alpha_up * 255.0).round().astype(np.uint8), mode="L")
            else:
                alpha_img = alpha.resize(result.size, RESAMPLE_BICUBIC)
            if alpha_img.size != result.size:
                alpha_img = alpha_img.resize(result.size, RESAMPLE_BICUBIC)
            result = result.convert("RGBA")
            result.putalpha(alpha_img)

        # The model has a fixed scale factor; resample if the user asked for
        # something else (e.g. 2x output from the 4x model).
        if not math.isclose(outscale, self.scale):
            target = (
                max(1, round(image.width * outscale)),
                max(1, round(image.height * outscale)),
            )
            if target != result.size:
                result = result.resize(target, RESAMPLE_LANCZOS)
        return result


def collect_inputs(paths: list[str], recursive: bool) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()

    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            candidates = sorted(p for p in path.glob(pattern) if p.is_file())
            for candidate in candidates:
                if candidate.suffix.lower() in SUPPORTED_SUFFIXES:
                    resolved = candidate.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        files.append(candidate)
        elif path.is_file():
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(path)
        else:
            print(f"warning: no such file or directory: {path}", file=sys.stderr)
    return files


def output_path_for(
    source: Path,
    root: Path | None,
    out_dir: Path,
    suffix: str,
    ext: str | None,
    has_alpha: bool,
) -> Path:
    relative = Path(source.name)
    if root is not None:
        try:
            relative = source.resolve().relative_to(root.resolve())
        except ValueError:
            relative = Path(source.name)

    target_ext = (ext or source.suffix.lstrip(".")).lower()
    if not target_ext:
        target_ext = "png"
    # JPEG and BMP cannot store transparency, so promote to PNG instead of
    # silently dropping the alpha channel.
    if has_alpha and f".{target_ext}" not in ALPHA_CAPABLE_SUFFIXES:
        target_ext = "png"

    return out_dir / relative.with_name(f"{relative.stem}{suffix}.{target_ext}")


def save_image(image: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    params: dict = {}
    if suffix in (".jpg", ".jpeg"):
        if image.mode != "RGB":
            image = image.convert("RGB")
        params = {"quality": quality, "subsampling": 0, "optimize": True}
    elif suffix == ".webp":
        params = {"quality": quality, "method": 6}
    elif suffix == ".png":
        params = {"compress_level": 6}
    image.save(path, **params)


def build_parser() -> argparse.ArgumentParser:
    model_help = "\n".join(f"  {name:<11} {spec.description}" for name, spec in MODELS.items())
    parser = argparse.ArgumentParser(
        prog="upscale.py",
        description="Batch upscale images with Real-ESRGAN (CUDA / Apple Silicon / CPU).",
        epilog=f"available models:\n{model_help}\n\n"
        "examples:\n"
        "  python upscale.py ./photos -o ./photos_4x\n"
        "  python upscale.py img1.jpg img2.png --model anime --scale 2\n"
        "  python upscale.py ./raw -r --tile 256 --ext png\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="PATH",
        help="input folder(s) and/or individual image files",
    )
    parser.add_argument(
        "-o", "--output", default="upscaled", metavar="DIR",
        help="output directory, created if missing (default: ./upscaled)",
    )
    parser.add_argument(
        "-m", "--model", default="general", choices=sorted(MODELS),
        help="model to use (default: general)",
    )
    parser.add_argument(
        "-s", "--scale", type=float, default=None, metavar="N",
        help="final scale factor; defaults to the model's native scale",
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="descend into subfolders and mirror the tree in the output dir",
    )
    parser.add_argument(
        "--suffix", default="_upscaled", metavar="STR",
        help="appended to each filename; pass '' to keep the original name",
    )
    parser.add_argument(
        "--ext", default=None, metavar="EXT",
        help="force an output format, e.g. png or webp (default: keep original)",
    )
    parser.add_argument(
        "--quality", type=int, default=95, metavar="N",
        help="JPEG/WebP quality, 1-100 (default: 95)",
    )
    parser.add_argument(
        "--device", default="auto", metavar="DEV",
        help="auto, cuda, cuda:1, mps or cpu (default: auto)",
    )
    parser.add_argument(
        "--tile", type=int, default=0, metavar="PX",
        help="tile size in pixels; 0 auto-tiles only when memory runs short",
    )
    parser.add_argument(
        "--tile-pad", type=int, default=32, metavar="PX",
        help="tile overlap used to hide seams (default: 32; lower is faster)",
    )
    parser.add_argument(
        "--fp32", action="store_true",
        help="disable half precision (CUDA defaults to fp16 for speed)",
    )
    parser.add_argument(
        "--alpha", choices=("bicubic", "model"), default="bicubic",
        help="how to upscale transparency (default: bicubic, faster)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="leave already-upscaled outputs alone; useful for resuming",
    )
    parser.add_argument(
        "--max-pixels", type=int, default=0, metavar="N",
        help="refuse inputs larger than N pixels; 0 means no limit",
    )
    parser.add_argument(
        "--model-dir", default=str(CACHE_DIR), metavar="DIR",
        help=f"where to cache downloaded weights (default: {CACHE_DIR})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="list what would be processed and exit",
    )
    return parser


def common_root(files: list[Path], inputs: list[str], recursive: bool) -> Path | None:
    """When recursing into a single folder, mirror its structure in the output."""
    if not recursive:
        return None
    dirs = [Path(p).expanduser() for p in inputs if Path(p).expanduser().is_dir()]
    if len(dirs) == 1 and all(str(f.resolve()).startswith(str(dirs[0].resolve())) for f in files):
        return dirs[0]
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not 1 <= args.quality <= 100:
        print("error: --quality must be between 1 and 100", file=sys.stderr)
        return 2
    if args.scale is not None and args.scale <= 0:
        print("error: --scale must be greater than 0", file=sys.stderr)
        return 2

    files = collect_inputs(args.inputs, args.recursive)
    if not files:
        print(
            "error: no supported images found "
            f"({', '.join(sorted(SUPPORTED_SUFFIXES))})",
            file=sys.stderr,
        )
        return 1

    spec = MODELS[args.model]
    outscale = args.scale if args.scale is not None else float(spec.scale)
    out_dir = Path(args.output).expanduser()
    root = common_root(files, args.inputs, args.recursive)

    if args.dry_run:
        print(f"{len(files)} image(s) would be written to {out_dir}/ at {outscale}x:")
        for path in files:
            print(f"  {path}")
        return 0

    device = resolve_device(args.device)
    half = (not args.fp32) and device.type == "cuda"

    print(f"device : {describe_device(device)}")
    print(f"model  : {spec.filename} ({spec.scale}x native) -> {outscale}x output")
    print(f"images : {len(files)}  ->  {out_dir}/")

    try:
        upscaler = Upscaler(
            spec=spec,
            device=device,
            half=half,
            tile=max(0, args.tile),
            tile_pad=max(0, args.tile_pad),
            cache_dir=Path(args.model_dir).expanduser(),
        )
    except Exception as exc:
        print(f"error: could not initialize model: {exc}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[Path, str]] = []
    skipped = 0
    succeeded = 0

    progress = tqdm(files, unit="img", desc="Upscaling", dynamic_ncols=True)
    for source in progress:
        progress.set_postfix_str(source.name[:40], refresh=False)
        try:
            with Image.open(source) as handle:
                handle.load()
                image = ImageOps.exif_transpose(handle)

                if args.max_pixels and image.width * image.height > args.max_pixels:
                    raise ValueError(
                        f"{image.width}x{image.height} exceeds --max-pixels ({args.max_pixels})"
                    )

                has_alpha = image.mode in ("RGBA", "LA", "PA") or (
                    image.mode == "P" and "transparency" in image.info
                )
                destination = output_path_for(
                    source, root, out_dir, args.suffix, args.ext, has_alpha
                )
                if destination.resolve() == source.resolve():
                    raise ValueError("output would overwrite the input; use --suffix or -o")
                if args.skip_existing and destination.exists():
                    skipped += 1
                    continue

                result = upscaler.upscale_image(image, outscale, args.alpha)
                save_image(result, destination, args.quality)
                succeeded += 1
        except KeyboardInterrupt:
            progress.close()
            print("\ninterrupted", file=sys.stderr)
            return 130
        except (UnidentifiedImageError, OSError, ValueError, RuntimeError) as exc:
            failures.append((source, f"{type(exc).__name__}: {exc}"))
            tqdm.write(f"failed: {source} -> {type(exc).__name__}: {exc}")
        except Exception as exc:  # unexpected: record the traceback and continue
            failures.append((source, f"{type(exc).__name__}: {exc}"))
            tqdm.write(f"failed: {source}\n{traceback.format_exc()}")
        finally:
            empty_device_cache(device)
    progress.close()

    print(f"\ndone: {succeeded} upscaled, {skipped} skipped, {len(failures)} failed")
    if failures:
        print("failures:", file=sys.stderr)
        for path, message in failures:
            print(f"  {path}: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
