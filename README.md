# PyTorch version of spatial transformer network

Ported from https://github.com/qassemoquab/stnbhwd according to pytorch tutorial.
Ported from https://github.com/fxia22/stn.pytorch  only for personal use
now support 3D spatial transformation, but have removed the GPU part.
# Build and test

```
cd script
./make.sh
python build.py
python test.py
```

There is a demo in `test_stn.py`

# Modules

`STN` is the spatial transformer module, it takes a `B*C*H*W*D` tensor and a `B*C*H*W*3` grid normalized to [-1,1] as an input and do bilinear sampling.

`AffineGridGen` takes a `B*3*4` matrix and generate an affine transformation grid.


# 3d-stn-pytorch
