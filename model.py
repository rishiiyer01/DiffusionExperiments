## going to create a spectral diffusion model
# the point of this is to experiment with alternative diffusion model architectures
#including explicit frequency autoregression

import torch
import torch.nn as nn
import torch.nn.functional as F








        





class ComplexSoftmax(nn.Module):
    def __init__(self, use_phase=True):
        super(ComplexSoftmax,self).__init__()
        # act can be either a function from nn.functional or a nn.Module if the
        # activation has learnable parameters
        self.act = nn.Softmax(dim=-1)
        self.use_phase = use_phase

    def forward(self, z):
        if self.use_phase:
            return self.act(torch.abs(z)) * torch.exp(1.j * torch.angle(z)) 
        else:
            return self.act(z.real) + 1.j * self.act(z.imag)

class SpectralTransformerLayer(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, spectral attention, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self,x):
        b,n,c,h,w=x.shape
        #the reason for n is that for each image in the batch, we have a set of noised images

        x_ft=torch.fft.rfft2(x) #does fft on last 2 dims
        



        


#this spectral conv2d layer simply implements a convolution in frequency space via linear layer
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform for QKV, spectral attention, and Inverse FFT.    
        """
        self.attention=nn.MultiheadAttention(in_channels, num_heads=8, batch_first=True,bias=False)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels,dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels,  dtype=torch.cfloat))
        self.weights5 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels,  dtype=torch.cfloat))
        self.softmax=ComplexSoftmax()

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,io->boxy", input, weights)
    
    def compl_mul(self,input,weights):
        return torch.einsum("bix,io->box", input, weights)
    

    def forward(self, x):
        b,c,h,w = x.shape
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        out_ft = \
            self.compl_mul2d(x_ft, self.weights1)
        out_ft = \
            self.compl_mul2d(x_ft, self.weights2)
        #print(out_ft.shape)
        #attention
        freqs=out_ft.reshape(b,c, h * (w//2+1))
        q=self.compl_mul(freqs,self.weights3)
        k=self.compl_mul(freqs,self.weights4)
        v=self.compl_mul(freqs,self.weights5)
        
        #attention
        print(q.shape,k.shape,v.shape)
        s=torch.einsum("bix,biy->bxy",q,k)
        s=self.softmax(s)
        print(s.shape)
        print(v.shape)
        out=torch.einsum("bxy,biy->bix",s,v)
        
        return out


model=SpectralConv2d(16,16,16,9)

x=torch.randn(2,16,16,16)
with torch.no_grad():
    out=model(x)
    print(out.shape)

