# functions/add.py
import torch
from torch.autograd import Function
import numpy as np


class AffineGridGenFunction(Function):
    def __init__(self, height, width, depth, lr=1):
        super(AffineGridGenFunction, self).__init__()
        self.lr = lr
        self.height, self.width, self.depth = height, width, depth
        self.grid = np.zeros( [4,self.height, self.width, self.depth], dtype=np.float32)
        x = np.repeat(np.expand_dims(np.repeat(
            np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / height), 0), repeats=width, axis=0).T, 0),
            repeats=depth, axis=0), 0), repeats=3, axis=0)
        y = np.ones((1,self.height,self.width,self.depth))
        self.grid[:3]=x
        self.grid[3] =y
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input1):
        self.input1 = input1
        output = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid

        for i in range(input1.size(0)):
                output = torch.bmm(self.batchgrid.view(-1, self.height*self.width*self.depth, 4), torch.transpose(input1, 1, 2)).view(-1, 3,self.height, self.width,self.depth)

        return output

    def backward(self, grad_output):

        grad_input1 = torch.zeros(self.input1.size())
        grad_input1 = torch.baddbmm(grad_input1, torch.transpose(grad_output.view(-1, self.height*self.width*self.depth, 3), 1,2), self.batchgrid.view(-1, self.height*self.width*self.depth,4))

        #print(grad_input1)
        return grad_input1
