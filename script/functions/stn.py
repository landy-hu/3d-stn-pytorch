# functions/add.py
import torch
from torch.autograd import Function
from _ext import my_lib
from cffi import FFI
ffi = FFI()

class STNFunction(Function):
    def forward(self, input1, input2):
        self.input1 = input1
        self.input2 = input2
        self.device_c = ffi.new("int *")
        print(self.device_c)
        output = torch.zeros(input1.size()[0], input2.size()[1], input2.size()[2], input1.size()[3])
        #print('decice %d' % torch.cuda.current_device())
        if input1.is_cuda:
            self.device = torch.cuda.current_device()
        else:
            self.device = -1
        self.device_c[0] = self.device
        if not input1.is_cuda:
            my_lib.BilinearSamplerBHWD_updateOutput(input1, input2, output)
        else:
            output = output.cuda(self.device)
            my_lib.BilinearSamplerBHWD_updateOutput_cuda(input1, input2, output, self.device_c)
        return output

    def backward(self, grad_output):
        print('hello!!!!!')
        grad_input1 = torch.zeros(self.input1.size())
        grad_input2 = torch.zeros(self.input2.size())
        #print('backward decice %d' % self.device)
        if not grad_output.is_cuda:
            my_lib.BilinearSamplerBHWD_updateGradInput(self.input1, self.input2, grad_input1, grad_input2, grad_output)
        else:
            grad_input1 = grad_input1.cuda(self.device)
            grad_input2 = grad_input2.cuda(self.device)
            print('BHWD:',grad_input1.size(),grad_input2.size())
            my_lib.BilinearSamplerBHWD_updateGradInput_cuda(self.input1, self.input2, grad_input1, grad_input2, grad_output, self.device_c)
        return grad_input1, grad_input2



class STNFunctionBCHW(Function):
    def forward(self, input1, input2):
        self.input1 = input1
        self.input2 = input2
        self.device_c = ffi.new("int *")
        output = torch.zeros(input1.size()[0], input1.size()[1], input2.size()[2], input2.size()[3],input2.size()[4])
        if input1.is_cuda:
            self.device = torch.cuda.current_device()
        else:
            self.device = -1
        self.device_c[0] = self.device
        print(input1.size(),input2.size(),output.size())
        if not input1.is_cuda:
            my_lib.BilinearSamplerBCHW_updateOutput(input1, input2, output)
        else:

            output = output.transpose(1,2).transpose(2,3).contiguous()
            input1 = input1.transpose(1,2).transpose(2,3).contiguous()
            input2 = input2.transpose(1,2).transpose(2,3).contiguous()
            #print(output.size(), input1.size(), input2.size())
            output = output.cuda(self.device)
            my_lib.BilinearSamplerBHWD_updateOutput_cuda(input1, input2, output, self.device_c)
            output = output.transpose(2,3).transpose(1,2)

        return output

    def backward(self, grad_output):
        # print('hello!!!1')
        grad_input1 = torch.zeros(self.input1.size())
        grad_input2 = torch.zeros(self.input2.size())
        # print(grad_input1.size(),grad_input2.size(),self.input1.size(),self.input2.size())
        #print('backward decice %d' % self.device)
        if not grad_output.is_cuda:
            # print(grad_output)
            # print('hello!!!2')
            my_lib.BilinearSamplerBCHW_updateGradInput(self.input1, self.input2, grad_input1, grad_input2, grad_output)
        else:
            # print('hello!!!2')
            grad_input1 = grad_input1.transpose(1,2).transpose(2,3).contiguous()
            # print('hello!!!3')
            grad_input2 = grad_input2.transpose(1,2).transpose(2,3).contiguous()
            grad_output = grad_output.transpose(1,2).transpose(2,3).contiguous()
            # print('hello!!!4')
            grad_input1 = grad_input1.cuda(self.device)
            grad_input2 = grad_input2.cuda(self.device)
            # print('hello!!!5')
            print('BHWD:',grad_input1.size(), grad_input2.size())
            my_lib.BilinearSamplerBHWD_updateGradInput_cuda(self.input1, self.input2, grad_input1, grad_input2, grad_output, self.device_c)
            # print('hello!!!6')
            grad_input1 = grad_input1.transpose(2,3).transpose(1,2)
            grad_input2 = grad_input2.transpose(2,3).transpose(1,2)
            # print('hello!!!7')
        return grad_input1, grad_input2
