## going to create a spectral diffusion model
# the point of this is to experiment with alternative diffusion model architectures
#including explicit frequency autoregression

import torch
import torch.nn as nn
import torch.nn.functional as F






class spectralModel(nn.Module):
    def __init__(self,in_channels,out_channels,num_blocks):
        super(spectralModel,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.num_blocks=num_blocks
        self.blocks=[spectralAuto(in_channels,out_channels) for _ in range(num_blocks)]
        self.final_linear=nn.Linear(out_channels,in_channels).to(torch.cfloat)

    
        
    def forward(self,x):
        xft=torch.fft.rfft2(x)
        for block in self.blocks:
            xft=block(xft)
        
        xft=xft.permute(0,2,3,1)

        xft=self.final_linear(xft)  
        xft=xft.permute(0,3,1,2)
        
        out=torch.fft.irfft2(xft)
        return out,xft
        








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

class feedForward(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(feedForward, self).__init__()
        self.weights1=nn.Parameter(torch.rand(in_channels, out_channels, dtype=torch.cfloat))
        self.rms_norm = nn.RMSNorm([in_channels])
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,io->boxy", input, weights)
    def forward(self, xft):
        b,c,hft,wft=xft.shape
        xft=xft.permute(0,2,3,1)
        xft=self.rms_norm(xft)
        xft=xft.permute(0,3,1,2)
        xft=self.compl_mul2d(xft,self.weights1)
        x=torch.fft.irfft2(xft)
        out=F.silu(x)
        out=torch.fft.rfft2(out)
        return out

class spectralAuto(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(spectralAuto, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform for QKV, spectral attention, and Inverse FFT.    
        """
        #self.attention=nn.MultiheadAttention(in_channels, num_heads=8, batch_first=True,bias=False)
        self.in_channels = in_channels
        self.out_channels = out_channels
        

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels,dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, dtype=torch.cfloat))
        self.feedForward=feedForward(out_channels, out_channels)
        self.transformerBlock=spectralTransformerBlock(out_channels, out_channels,num_heads=8)
    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,io->boxy", input, weights)
    
    def compl_mul(self,input,weights):
        return torch.einsum("bix,io->box", input, weights)

    def forward(self, xft):
       # b,c,h,w = x.shape
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        #x_ft = torch.fft.rfft2(x)

        out_ft = \
            self.compl_mul2d(xft, self.weights1) #linear layer on channel (doesn't mix frequencies)
        x=torch.fft.irfft2(out_ft)
        x=F.silu(x)

        xft=torch.fft.rfft2(x)
        xft=self.transformerBlock(xft)
        
        out=self.feedForward(xft)+xft

        #as of right now this is just a single head single block model.

        #out=torch.fft.irfft2(out)
        
        return out

class spectralTransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8):
        super(spectralTransformerBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads

        assert self.head_dim * num_heads == out_channels, "out_channels must be divisible by num_heads"

        self.scale = 1 / (self.head_dim ** 0.5)

        # Weights for Q, K, V
        self.weights_q = nn.Parameter(torch.rand(in_channels, out_channels, dtype=torch.cfloat))
        self.weights_k = nn.Parameter(torch.rand(in_channels, out_channels, dtype=torch.cfloat))
        self.weights_v = nn.Parameter(torch.rand(in_channels, out_channels, dtype=torch.cfloat))

        # Output projection
        self.weights_o = nn.Parameter(torch.rand(out_channels, out_channels, dtype=torch.cfloat))

        self.softmax = ComplexSoftmax()
        self.rms_norm = nn.RMSNorm([in_channels])

    def generate_causal_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        
        return mask

    def split_heads(self, x):
        b, c, hw = x.shape
        return x.view(b, self.num_heads, self.head_dim, hw)
    
    def combine_heads(self, x):
        b, _, h, hw = x.shape
        return x.reshape(b, h * self.num_heads, hw)

    def forward(self, xfft):
        b, c, hft, wft = xfft.shape
        hw = hft * wft

        # Normalize and reshape
        xfft=xfft.permute(0,2,3,1)
        out_ft = self.rms_norm(xfft)
        xfft=xfft.permute(0,3,1,2)
        out_ft=out_ft.permute(0,3,1,2)
        freqs = out_ft.reshape(b, c, hw)

        # Compute Q, K, V
        q = torch.einsum("bix,io->box", freqs, self.weights_q)
        k = torch.einsum("bix,io->box", freqs, self.weights_k)
        v = torch.einsum("bix,io->box", freqs, self.weights_v)
        

        # Split heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Compute attention scores
        s = torch.einsum("bnix,bniy->bnxy", q, k) * self.scale

        # Apply causal mask
        causal_mask = self.generate_causal_mask(hw).to(s.device)
        
        s = s.masked_fill(causal_mask, float('-inf'))

        # Apply softmax
        s = self.softmax(s)

        # Apply attention to values
        out = torch.einsum("bnxy,bniy->bnix", s, v)

        # Combine heads
        out = self.combine_heads(out)

        # Apply output projection
        out = torch.einsum("bix,io->box", out, self.weights_o)

        # Reshape and add residual connection
        out = out.reshape(b, c, hft, wft) + xfft

        return out


model=spectralModel(16,16,4)

x=torch.randn(2,16,16,16)
out=model(x)
print(out[0].shape)