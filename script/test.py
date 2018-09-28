from __future__ import print_function

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from modules.stn import STN
from modules.gridgen import AffineGridGen, CylinderGridGen, CylinderGridGenV2, DenseAffine3DGridGen, DenseAffine3DGridGen_rotate

import time

nframes = 5
height = 64
width = 64
channels = 4
depth =64

inputImages = torch.zeros(nframes,channels, height, width, depth)
x = np.repeat(np.expand_dims(np.repeat(np.expand_dims(np.repeat(np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/height), 0), repeats =width, axis = 0).T, 0),repeats=depth,axis=0),0),repeats=3,axis=0),0),repeats=nframes,axis=0)
grids = torch.from_numpy(x.astype(np.float32))
input1, input2 = Variable(inputImages, requires_grad=True), Variable(grids, requires_grad=True)
input1.data.uniform_()
input2.data.uniform_(-1,1)

s2 = STN(layout = 'BCHW')
start = time.time()
out = s2(input1, input2)
print('forward:',out.size(), 'time:', time.time() - start)
start = time.time()
out.backward(input1.data)
print('backward',input1.grad.size(), 'time:', time.time() - start)

