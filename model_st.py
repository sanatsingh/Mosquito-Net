import torch.nn as nn
import torch.nn.functional as F

class MosquitoNet(nn.Module):
    ''' Original MosquitoNet Model '''
    def __init__(self):
        super(MosquitoNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
            
        self.fc1 = nn.Linear(64*15*15, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        self.drop = nn.Dropout2d(0.2)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)    # flatten out a input for Dense Layer
        out = self.fc1(out)
        out = F.relu(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.drop(out)
        out = self.fc3(out)
        
        return out

class Mish(nn.Module):
    '''
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    '''
    @torch.jit.script
    def mish(input):
        return input * torch.tanh(F.softplus(input))

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return self.mish(input)

class MosquitoNet_Mish(nn.Module):
    ''' Another version of MosquitoNet using Mish activation function, outperforms Original Version'''
    def __init__(self):
        super(MosquitoNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
            
        self.fc1 = nn.Linear(64*15*15, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 2)
        self.drop = nn.Dropout2d(0.2)
        self.m=Mish()
    
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)    # flatten out a input for Dense Layer
        out = self.fc1(out)
        out = self.m(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.m(out)
        out = self.drop(out)
        out = self.fc3(out)
        
        return out


class DepthwiseSepConv(nn.Module):
    def __init__(self, inch, ch, kernel_size=3, stride=1, dilation=1, bias=True, BatchNorm=True):
        '''
        DepthwiseSepConv Module implemented in PyTorch. Divided in 2 parts:-
        1) Depth Wise Convolution : using nn.Conv2d layer with groups parameter each kernel is passed for a single layer of input
        2) Point Wise Convolution : 1x1 Convolution implemented using nn.Conv2d layer

        BatchNorm (optional) available, requires to pass the function/module to be used.

        '''
        super(DepthwiseSepConv, self).__init__()
        self.depthconv = nn.Conv2d(
            inch, inch, kernel_size, stride, 0, dilation, groups=inch, bias=bias)
        self.btn = nn.BatchNorm2d(inch)
        self.pointconv = nn.Conv2d(inch, ch, 1, 1, 0, 1, 1, bias=bias)
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.BatchNorm = BatchNorm

    def padding_fix(self, x, kernel_size, dilation):
        kernel_size_effective = kernel_size + \
            (kernel_size - 1) * (dilation - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = F.pad(x, (pad_beg, pad_end, pad_beg, pad_end))
        return padded_inputs

    def forward(self, x):
        x = self.padding_fix(x, self.kernel_size, self.dilation)
        x = self.depthconv(x)
        if not self.BatchNorm:
            x = self.btn(x)
        x = self.pointconv(x)
        return x

class MosquitoNet_V2(nn.Module):
    ''' Another version of MosquitoNet using Mish activation function, outperforms Original Version'''

    def __init__(self):
        super(MosquitoNet_V2, self).__init__()

        self.layer1 = nn.Sequential(
            DepthwiseSepConv(3, 16, kernel_size=5, stride=1),
            Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            DepthwiseSepConv(16, 32, kernel_size=3, stride=1),
            Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            DepthwiseSepConv(32, 64, kernel_size=3, stride=1),
            Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            DepthwiseSepConv(64, 128, kernel_size=3, stride=1),
            Mish(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(128*7*7, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 2)
        self.drop = nn.Dropout2d(0.2)
        self.m = Mish()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # flatten out a input for Dense Layer
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.m(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.m(out)
        out = self.drop(out)
        out = self.fc3(out)

        return out
