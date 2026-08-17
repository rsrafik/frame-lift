# Batch Image Upscaler

Real-ESRGAN batch upscaler, plus a temporal stabilizer for video frame sequences. Point it at a folder and it writes upscaled copies to an output directory, using CUDA, Apple Silicon (MPS) or CPU automatically.

The model architectures live inside `upscale.py`, so there's no dependency on `basicsr` or the `realesrgan` package (both break on recent torchvision and Python 3.12+). Weights download on first run.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On an NVIDIA GPU, install torch from the CUDA index first so you don't get the CPU-only wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Confirm the GPU was found:

```bash
python -c "import torch; print('cuda', torch.cuda.is_available(), '| mps', torch.backends.mps.is_available())"
```

## Usage

```bash
python upscale.py ./photos -o ./photos_4x                    # whole folder, 4x
python upscale.py a.jpg b.png --model anime --scale 2        # single files, 2x
python upscale.py ./raw -o ./out -r --ext png --suffix ''    # recurse, keep filenames
python upscale.py ./photos -o ./out --skip-existing          # resume an interrupted run
python upscale.py ./photos --dry-run                         # preview, process nothing
```

Handles `.png` `.jpg` `.jpeg` `.webp` `.bmp` `.tif` `.tiff`. A corrupt file is logged and skipped; the batch continues. Exit code is `0` if everything succeeded, `1` if anything failed.

| Flag | Default | Description |
|---|---|---|
| `-o, --output DIR` | `upscaled` | Output directory |
| `-m, --model NAME` | `general` | See models below |
| `-s, --scale N` | model native | Final scale factor |
| `-r, --recursive` | off | Descend into subfolders, mirror the tree |
| `--suffix STR` | `_upscaled` | Pass `''` to keep original filenames |
| `--ext EXT` | keep original | Force output format, e.g. `png` |
| `--quality N` | `95` | JPEG/WebP quality |
| `--device DEV` | `auto` | `auto`, `cuda`, `cuda:1`, `mps`, `cpu` |
| `--tile PX` | `0` | Tile size; `0` tiles only when memory runs short |
| `--tile-pad PX` | `32` | Tile overlap; lower is faster |
| `--fp32` | off | Disable half precision (CUDA defaults to fp16) |
| `--alpha MODE` | `bicubic` | `bicubic` or `model` for transparency |
| `--skip-existing` | off | Leave existing outputs alone |
| `--max-pixels N` | `0` | Reject inputs above N pixels |
| `--model-dir DIR` | `~/.cache/upscale-models` | Weight cache |
| `--dry-run` | off | List and exit |

## Models

Weights download on first use to `~/.cache/upscale-models` (override with `--model-dir` or `UPSCALE_MODEL_DIR`), from the official Real-ESRGAN releases. Offline machines can be seeded by dropping the `.pth` file in that directory.

| `--model` | Weights | Scale | Size | For |
|---|---|---|---|---|
| `general` (default) | `RealESRGAN_x4plus.pth` | 4x | 64 MB | Photos, general images |
| `general-x2` | `RealESRGAN_x2plus.pth` | 2x | 64 MB | 2x output, less memory |
| `anime` | `RealESRGAN_x4plus_anime_6B.pth` | 4x | 17 MB | Anime, illustration, line art |
| `fast` | `realesr-general-x4v3.pth` | 4x | 5 MB | CPU-only, or large batches |

## Video frames

Real-ESRGAN upscales each image independently, so across a sequence it invents slightly different fine texture every frame, which reads as shimmer. Geometry never moves — the network is fully convolutional and spatially aligned — but the texture does.

`stabilize.py` removes it as a post-pass. It detects motion on the **original** frames, which carry no hallucinated detail, replaces static pixels with the median of the upscaled frames in the window, and passes moving pixels through untouched:

```bash
python stabilize.py Frames_4x --reference Frames -o Frames_stable
```

| Flag | Default | Description |
|---|---|---|
| `--reference DIR` | required | The original frames, used only to detect motion |
| `-o, --output DIR` | `stabilized` | Output folder |
| `--window N` | `3` | Temporal window, odd. Larger is steadier and slower |
| `--threshold V` | `3.0` | Reference change (0-255) counted as fully static |
| `--fade F` | `2.5` | Stabilization fades out by `threshold * F` |
| `--ref-blur R` | `1.0` | Blur on the motion reference, to ignore compression noise |
| `--strength F` | `1.0` | How much to apply in static areas, 0-1 |
| `--feather PX` | `2` | Blur radius on the motion mask |

If flicker survives, raise `--window` to 5. If moving areas smear, lower `--threshold` or `--strength`. Frames must sort in sequence order; `frame_2` sorts before `frame_10` regardless of padding. Stabilization is separate from upscaling, so you can retune it without repeating the slow pass.

## Measurements

Apple M3 Max, 512x384 to 2048x1536, fp32:

| Model | MPS | CPU |
|---|---|---|
| `general` | 0.91 s/img | 18.0 s/img |
| `fast` | 0.07 s/img | 0.93 s/img |

Restoring a 128x96 image to 512x384 against the original: bicubic **18.26 dB** PSNR, Real-ESRGAN `general` **22.27 dB**.

Temporal change in static regions across a 192-frame sequence, against a Lanczos resize of the same frames: raw upscale **1.31x**, after `stabilize.py` **0.94x**, costing 2.0% edge sharpness. Below 1.00x means steadier than a plain resize.

## Notes

- Transparency is preserved. JPEG and BMP can't store alpha, so images with transparency are written as PNG even under `--ext jpg`, rather than being silently flattened.
- EXIF orientation is applied before upscaling.
- Writing an output over its own input is blocked.
- Tiling isn't bit-identical to whole-image processing, since a tile can't see past its padding. At `--tile-pad 32` the mean deviation was 0.27/255, with under 0.001% of pixels off by more than 16/255.
- On OOM the upscaler retries at 512px tiles, then 256, then 128. `--tile 256` skips the failed first attempt if you already know the images are large.

## Troubleshooting

**`CERTIFICATE_VERIFY_FAILED`** — macOS framework Python often ships without a CA store. `pip install certifi` fixes it.

**`torch.cuda.is_available()` is False** — you have the CPU-only wheel; reinstall from the CUDA index above.

**Out of memory** — try `--tile 128 --fp32`, or `--model general-x2` / `--model fast`.

**MPS output looks wrong** — some PyTorch releases have MPS bugs. Upgrade, or use `--device cpu`.

## License

MIT (see `LICENSE`).

The network architectures are reimplemented from BasicSR (Apache-2.0) and
Real-ESRGAN (BSD-3-Clause); the pretrained weights are BSD-3-Clause and are
downloaded at runtime rather than redistributed here. Attribution and the terms
those licenses require are in [NOTICE.md](NOTICE.md) — keep that file if you
fork this.
