# Third-party notices

This project is MIT licensed (see `LICENSE`). It reimplements neural network
architectures from two upstream projects and downloads pretrained weights from a
third. Their terms are below.

## RRDBNet, RRDB, ResidualDenseBlock

Reimplemented from **BasicSR**, `basicsr/archs/rrdbnet_arch.py`.

- Source: https://github.com/XPixelGroup/BasicSR
- License: Apache License 2.0
- Modifications: rewritten as standalone PyTorch modules with type annotations,
  with the BasicSR registry, config plumbing and `make_layer` helper removed.
  Layer names, channel counts and forward-pass arithmetic are unchanged so the
  official checkpoints load without remapping.

Apache-2.0 requires that this notice, the copyright attribution above, and a
statement of changes accompany redistribution. Full license text:
https://www.apache.org/licenses/LICENSE-2.0

## SRVGGNetCompact

Reimplemented from **Real-ESRGAN**, `realesrgan/archs/srvgg_arch.py`.

- Source: https://github.com/xinntao/Real-ESRGAN
- Copyright (c) 2021, Xintao Wang
- License: BSD 3-Clause
- Modifications: the activation-type branch was reduced to PReLU only, which is
  what the distributed `realesr-general-x4v3` checkpoint uses.

BSD-3-Clause requires that the copyright notice, this list of conditions and the
disclaimer be retained in redistributions.

## Pretrained weights

`RealESRGAN_x4plus.pth`, `RealESRGAN_x2plus.pth`, `RealESRGAN_x4plus_anime_6B.pth`
and `realesr-general-x4v3.pth` are downloaded at runtime from the Real-ESRGAN
releases and are **not** redistributed in this repository.

- Source: https://github.com/xinntao/Real-ESRGAN/releases
- License: BSD 3-Clause

If you vendor the `.pth` files into a fork or ship them inside an application,
carry the Real-ESRGAN license with them.
