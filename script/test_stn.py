from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from modules.stn import STN
from modules.gridgen import  AffineGridGen

nframes = 5
height = 64
width = 64
channels = 1
depth =64
##generate the 3D voxel
inputvoxel = np.random.rand(nframes,channels, height, width, depth)
inputvoxel = inputvoxel.astype(np.float32)
inputvoxel = torch.from_numpy(inputvoxel)

target = np.random.rand(nframes,channels, height, width, depth)
target = target.astype(np.float32)
target = torch.from_numpy(target)
#initialization stn
s = STN(layout = 'BCHW')
g = AffineGridGen(64,64,64)
# set the samll transformation
input = Variable(torch.from_numpy(np.array([[[0.9, 0.01, 0, 0], [0.01, 0.9, 0, 0],[0, 0.01, 0.9, 0]]], dtype=np.float32)), requires_grad = True)
# get the volume after transformation
out = g(input)
#
out = out.repeat(nframes,1,1,1,1)
input1 = Variable(inputvoxel, requires_grad = True)
res = s(input1, out)
target = Variable(target, requires_grad = False)
crt = nn.MSELoss()
loss = crt(res, target)
loss.backward()
print('hello')

