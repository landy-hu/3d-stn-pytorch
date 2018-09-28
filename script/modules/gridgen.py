from torch.nn.modules.module import Module
import torch
from torch.autograd import Variable
import numpy as np
from functions.gridgen import AffineGridGenFunction

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)


class AffineGridGen(Module):
    def __init__(self, height, width, depth, lr = 1, aux_loss = False):
        super(AffineGridGen, self).__init__()
        self.height, self.width, self.depth = height, width, depth
        self.aux_loss = aux_loss
        self.f = AffineGridGenFunction(self.height, self.width,self.depth, lr=lr)
        self.lr = lr
    def forward(self, input):
        if not self.aux_loss:
            return self.f(input)
        else:
            identity = torch.from_numpy(np.array([[1,0,0,0], [0,1,0,0],[0,0,1,0]], dtype=np.float32))
            batch_identity = torch.zeros([input.size(0), 3,4])
            for i in range(input.size(0)):
                batch_identity[i] = identity
            batch_identity = Variable(batch_identity)
            loss = torch.mul(input - batch_identity, input - batch_identity)
            loss = torch.sum(loss)
            loss = torch.sum(loss)

            return self.f(input), loss.view(-1,1)

