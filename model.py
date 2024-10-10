## going to create a spectral diffusion model
# the point of this is to experiment with alternative diffusion model architectures
#including explicit frequency autoregression

import torch
import torch.nn as nn
import torch.nn.functional as F



class PositionalEncoding(nn.Module):
    def __init__(self,d_model):
        super(PositionalEncoding,self).__init__()
        
        self.linear=nn.Linear(1,d_model,bias=False)
    def forward(self,image):
        b,c,h,w=image.shape
        xfreqs=torch.fft.fftfreq(h,dtype=torch.float)
        yfreqs=torch.fft.fftfreq(w,dtype=torch.float)
        freqspace=torch.einsum('i,j->ij',xfreqs,yfreqs)
        freqs=freqspace.unsqueeze(0)
        x=self.linear(freqs)

        return x

        



class spectralModel(nn.Module):
    def __init__(self,in_channels,out_channels,num_blocks):
        super(spectralModel,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.num_blocks=num_blocks
        self.blocks=[spectralAuto(in_channels,out_channels) for _ in range(num_blocks)]
        self.final_linear=nn.Linear(out_channels,in_channels).to(torch.cfloat)
        self.input_linear=nn.Linear(in_channels,out_channels).to(torch.cfloat)

        
    def forward(self,x):
        
        xft=torch.fft.rfft2(x,norm='ortho')
        xft=xft.permute(0,2,3,1)
        xft=self.input_linear(xft)
        xft=xft.permute(0,3,1,2)
        x=torch.fft.irfft2(xft,norm='ortho')
        
        for block in self.blocks:
            x=block(x)
        
        
        xft=torch.fft.rfft2(x,norm='ortho')
        xft=xft.permute(0,2,3,1)

        xft=self.final_linear(xft)  
        xft=xft.permute(0,3,1,2)
        
        out=torch.fft.irfft2(xft,norm='ortho')
        return out,xft
        








class ComplexSoftmax(nn.Module):
    def __init__(self, use_phase=False):
        super(ComplexSoftmax,self).__init__()
        # act can be either a function from nn.functional or a nn.Module if the
        # activation has learnable parameters
        self.act = nn.Softmax(dim=1)
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
    def forward(self, x):
        x=x.permute(0,2,3,1)
        
        x=self.rms_norm(x)
        x=x.permute(0,3,1,2)
        xft=torch.fft.rfft2(x,norm='ortho')
        b,c,hft,wft=xft.shape
        xft=self.compl_mul2d(xft,self.weights1)
        x=torch.fft.irfft2(xft,norm='ortho')
        out=F.silu(x)
        #out=torch.fft.rfft2(out)
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

    def forward(self, x):
       # b,c,h,w = x.shape
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        #x_ft = torch.fft.rfft2(x)

        #out_ft = \
        #    self.compl_mul2d(xft, self.weights1) #linear layer on channel (doesn't mix frequencies)
        #x=torch.fft.irfft2(out_ft)
        #x=F.silu(x)
        xtf=self.transformerBlock(x)
        
        out=self.feedForward(xtf)+xtf

        

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
        #self.pos_encoding=PositionalEncoding(out_channels,x)
        # Output projection
        self.weights_o = nn.Parameter(torch.rand(out_channels, out_channels, dtype=torch.cfloat))

        self.softmax = nn.Softmax(dim=1)
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

    def forward(self, x):
       
        
        x=x.permute(0,2,3,1)
        x=self.rms_norm(x)
        
        x=x.permute(0,3,1,2)
        #pos_enc=self.pos_encoding(x)
        xfft=torch.fft.rfft2(x,norm='ortho')
        b, c, hft, wft = xfft.shape
        hw = hft * wft

        ##need to normalize xfft

        
        
        
        
        freqs = xfft.reshape(b, c, hw)
        #freqs+=pos_enc #adding positional encoding
        # Compute Q, K, V
        q = torch.einsum("bix,io->box", freqs, self.weights_q)
        k = torch.einsum("bix,io->box", freqs, self.weights_k)
        v = torch.einsum("bix,io->box", freqs, self.weights_v)
        

        # Split heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        # Compute attention scores
        s = torch.einsum("bnix,bniy->bnxy", q.conj(), k) * self.scale 
        
        # it should be noted here that the scores here are complex valued, to associate them with real probabilities we take the square of the absolute value or softmax the absolute value
        s=torch.abs(s)
        
        
        # Apply causal mask
        causal_mask = self.generate_causal_mask(hw).to(s.device)
        
        
        
        
        # Apply softmax
        s = self.softmax(s)
        
        s = s.masked_fill(causal_mask, 0).to(torch.cfloat)

        

        # Apply attention to values
        out = torch.einsum("bnxy,bniy->bnix", s, v)
        
        # Combine heads
        out = self.combine_heads(out)

        # Apply output projection
        out = torch.einsum("bix,io->box", out, self.weights_o)

        # Reshape and add residual connection
        out = out.reshape(b, c, hft, wft) + xfft

        out=torch.fft.irfft2(out,norm='ortho')

        return out

model=spectralModel(16,16,1)

