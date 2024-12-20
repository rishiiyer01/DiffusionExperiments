## going to create a spectral diffusion model
# the point of this is to experiment with alternative diffusion model architectures
#including explicit frequency autoregression

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import inf


class PositionalEncoding(nn.Module):
    def __init__(self,d_model):
        super(PositionalEncoding,self).__init__()
        
        self.linear=nn.Linear(2,d_model,bias=True).to(torch.float) #bias for translation
    def forward(self,image):
        b,c,h,w=image.shape
        xfreqs=torch.fft.fftfreq(h,dtype=torch.float)
        yfreqs=torch.fft.rfftfreq(w,dtype=torch.float)
        xfreqgrid,yfreqgrid=torch.meshgrid(xfreqs,yfreqs, indexing='ij') #full spectrum,real spectrum
        xfreqgrid=xfreqgrid.expand(b,-1,-1).unsqueeze(1)
        yfreqgrid=yfreqgrid.expand(b,-1,-1).unsqueeze(1)
        freqgrid=torch.cat((xfreqgrid,yfreqgrid),dim=1)
        freqgrid=freqgrid.permute(0,2,3,1)
        freqgrid=self.linear(freqgrid)
        freqgrid=freqgrid.permute(0,3,1,2)
        
        x=F.normalize(freqgrid,dim=1)
        x=x.to(torch.cfloat).to('cuda')
        
        #print(x.shape, 'pos_enc') 
        
        return x

        



class spectralModel(nn.Module):
    def __init__(self,in_channels,out_channels,num_blocks):
        super(spectralModel,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.num_blocks=num_blocks
        self.blocks=[spectralAuto(out_channels,out_channels) for _ in range(num_blocks)]
        self.abs_linear=nn.Linear(out_channels,3*256,dtype=torch.float,bias=True)
        
        self.phase_linear=nn.Linear(out_channels,3*256,dtype=torch.float,bias=True) #256 is the finite scalar quantization vocab size
        
        self.input_linear=nn.Linear(in_channels,out_channels,dtype=torch.cfloat,bias=True)
        self.softmax = nn.Softmax(dim=-1)
        
        self.dc_linear=nn.Linear(1,out_channels,dtype=torch.cfloat,bias=True)


        
    def create_index_maps(self,height, width):
        indices = []
        flat_indices = []
        index = 0
        for sum_indices in range(height + width - 1):
            for i in range(min(sum_indices + 1, height)):
                j = sum_indices - i
                if j < width:
                    indices.append((i, j))
                    flat_indices.append(index)
                    index += 1
        
        indices = torch.tensor(indices)
        flat_indices = torch.tensor(flat_indices)
        return indices, flat_indices
        
    def custom_flatten(self,tensor):
        batch, vocab,channel, height, width = tensor.shape
        device = tensor.device
        
        indices, flat_indices = self.create_index_maps(height, width)
        indices = indices.to(device)
        flat_indices = flat_indices.to(device)
        
        flattened = torch.zeros(batch, vocab,channel, height * width, device=device).to(tensor.dtype)
        flattened[:, :,:, flat_indices] = tensor[:, :,:, indices[:, 0], indices[:, 1]]
        
        
        return flattened

    def custom_unflatten(self,flattened, height, width):
        batch, channel, _ = flattened.shape
        device = flattened.device
        
        indices, flat_indices = self.create_index_maps(height, width)
        indices = indices.to(device)
        flat_indices = flat_indices.to(device)
        
        unflattened = torch.zeros(batch, channel, height, width, device=device).to(flattened.dtype)
        unflattened[:, :, indices[:, 0], indices[:, 1]] = flattened[:, :, flat_indices]
        
        return unflattened

        
    def forward(self,x):

        b, c, h, w = x.shape
        #we randomly sample the first frequency from an imaginary gaussian
        first_freq=torch.randn(b,1,1,dtype=torch.cfloat).to('cuda')
        first_freq=self.dc_linear(first_freq)
        first_freq=first_freq.permute(0,2,1)
        
        
        
        xft=torch.fft.rfft2(x,norm='ortho')
        b1,c1,h1,w1=xft.shape
        xft=xft.permute(0,2,3,1)
        xft=self.input_linear(xft)
        xft=xft.permute(0,3,1,2)
        xft=xft.unsqueeze(dim=1)
        xft=self.custom_flatten(xft)
        xft=xft.squeeze(dim=1)
        xft=torch.cat((first_freq,xft[:,:,:-1]),dim=-1)
        xft=self.custom_unflatten(xft,h1,w1)
        
        #concat first freq to everything else
        


        


        
        
        x=torch.fft.irfft2(xft,norm='ortho')
        x=F.silu(x)
        
        for block in self.blocks:
            x=block(x)
            #print(torch.linalg.norm(x[0,0,:,:]))
            
        #print(torch.linalg.norm(x))   
        
        
        xft=torch.fft.rfft2(x,norm='ortho')
        xft=xft.permute(0,2,3,1)
        #print(xft,'xft')
        
        xft_abs=self.abs_linear(torch.abs(xft)) #b,h,w,c=70*3
        xft_phase=self.phase_linear(torch.abs(xft)) #b,h,w,c=100*3
        
        xft_r_phase, xft_g_phase, xft_b_phase = torch.split(xft_phase, 256, dim=-1)  
        xft_r_abs, xft_g_abs, xft_b_abs = torch.split(xft_abs, 256, dim=-1)   
        xft_phase=torch.stack((xft_r_phase, xft_g_phase, xft_b_phase),dim=-1) #b,h,w,c=3,vocab_size
        xft_abs=torch.stack((xft_r_abs, xft_g_abs, xft_b_abs),dim=-1) #b,h,w,c,vocab
        #xft_abs=self.softmax(xft_abs)
        #xft_phase=self.softmax(xft_phase)
        
        xft_abs=xft_abs.permute(0,4,3,1,2) #b,vocab,c,h,w
        xft_phase=xft_phase.permute(0,4,3,1,2) #b,vocab,c,h,w
        #out=torch.fft.irfft2(xft,norm='ortho') #b,vocab,c,x,y
        xftflat_phase=self.custom_flatten(xft_phase) #b,vocab,c,x*y
        xftflat_abs=self.custom_flatten(xft_abs)
        print(xftflat_phase.shape)
        return xftflat_abs,xftflat_phase
        








class ComplexSoftmax(nn.Module):
    def __init__(self, use_phase=True):
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
    def __init__(self, in_channels, out_channels,device='cuda'):
        super(feedForward, self).__init__()
        self.weights1=nn.Parameter(torch.rand(in_channels, out_channels, dtype=torch.cfloat)).to(device)
        self.rms_norm = nn.RMSNorm([in_channels]).to(device)
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,io->boxy", input, weights)
    def forward(self, x):
        #x=x.permute(0,2,3,1)
        #x=x/(torch.linalg.norm(x,dim=-1).unsqueeze(-1))
        #x=self.rms_norm(x)
        #x=x.permute(0,3,1,2)
        x=F.normalize(x)
        xft=torch.fft.rfft2(x,norm='ortho')
        b,c,hft,wft=xft.shape
        xft=self.compl_mul2d(xft,self.weights1)
        x=torch.fft.irfft2(xft,norm='ortho')
        #out=F.normalize(x)
        out=F.silu(x)
        #print(torch.linalg.norm(out))
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
        
        self.feedForward=feedForward(out_channels, out_channels)
        self.transformerBlock=spectralTransformerBlock(out_channels, out_channels,num_heads=16)
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
    def __init__(self, in_channels, out_channels, num_heads=12,device='cuda'):
        super(spectralTransformerBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads

        assert self.head_dim * num_heads == out_channels, "out_channels must be divisible by num_heads"

        self.scale = 1 / (self.head_dim)

        # Weights for Q, K, V
        self.weights_q = nn.Parameter(torch.rand(in_channels, out_channels, dtype=torch.cfloat)).to(device)
        self.weights_k = nn.Parameter(torch.rand(in_channels, out_channels, dtype=torch.cfloat)).to(device)
        self.weights_v = nn.Parameter(torch.rand(in_channels, out_channels, dtype=torch.cfloat)).to(device)
        self.pos_encoding=PositionalEncoding(out_channels)
        # Output projection
        self.weights_o = nn.Parameter(torch.rand(out_channels, out_channels, dtype=torch.cfloat)).to(device)

        self.softmax = nn.Softmax(dim=-1)
        self.rms_norm = nn.RMSNorm([in_channels]).to(device)
        self.dropout = nn.Dropout(0.1)

    def generate_causal_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        
        return mask

    def split_heads(self, x):
        b, c, hw = x.shape
        return x.view(b, self.num_heads, self.head_dim, hw)
    
    def combine_heads(self, x):
        b, _, h, hw = x.shape
        return x.reshape(b, h * self.num_heads, hw)
        
    def create_index_maps(self,height, width):
        indices = []
        flat_indices = []
        index = 0
        for sum_indices in range(height + width - 1):
            for i in range(min(sum_indices + 1, height)):
                j = sum_indices - i
                if j < width:
                    indices.append((i, j))
                    flat_indices.append(index)
                    index += 1
        
        indices = torch.tensor(indices)
        flat_indices = torch.tensor(flat_indices)
        return indices, flat_indices
    
    def custom_flatten(self,tensor):
        batch, channel, height, width = tensor.shape
        device = tensor.device
        
        indices, flat_indices = self.create_index_maps(height, width)
        indices = indices.to(device)
        flat_indices = flat_indices.to(device)
        
        flattened = torch.zeros(batch, channel, height * width, device=device).to(torch.cfloat)
        flattened[:, :, flat_indices] = tensor[:, :, indices[:, 0], indices[:, 1]]
        
        return flattened
    
    def custom_unflatten(self,flattened, height, width):
        batch, channel, _ = flattened.shape
        device = flattened.device
        
        indices, flat_indices = self.create_index_maps(height, width)
        indices = indices.to(device)
        flat_indices = flat_indices.to(device)
        
        unflattened = torch.zeros(batch, channel, height, width, device=device).to(torch.cfloat)
        unflattened[:, :, indices[:, 0], indices[:, 1]] = flattened[:, :, flat_indices]
        
        return unflattened

    def forward(self, x):
       
        
        #x=x.permute(0,2,3,1)
        
        #x=self.rms_norm(x)
        
        #x=x/(torch.linalg.norm(x,dim=-1).unsqueeze(-1))
        
        #x=x.permute(0,3,1,2)
        x=F.normalize(x)
        pos_enc=self.pos_encoding(x)
        xfft=torch.fft.rfft2(x,norm='ortho')
        #print(torch.linalg.norm(xfft))
        b, c, hft, wft = xfft.shape
        hw = hft * wft

        ##need to normalize xfft

        
        #print(xfft.shape)
        xfft+=pos_enc #adding positional encoding
        
        #freqs = xfft.reshape(b, c, hw)
        freqs=self.custom_flatten(xfft)
        
        
        
        # Compute Q, K, V
        q = F.normalize(torch.einsum("bix,io->box", freqs, self.weights_q))
        k = F.normalize(torch.einsum("bix,io->box", freqs, self.weights_k))
        v = F.normalize(torch.einsum("bix,io->box", freqs, self.weights_v))
        

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
        
        
        s = s.masked_fill(causal_mask, -inf)
        
        # Apply softmax
        s = self.softmax(s).to(torch.cfloat)
        
        
        

        

        # Apply attention to values
        out = torch.einsum("bnxy,bniy->bnix", s, v)
        
        # Combine heads
        out = self.combine_heads(out)

        # Apply output projection
        out = torch.einsum("bix,io->box", out, self.weights_o)
        
        # Reshape and add residual connection
        #out = out.reshape(b, c, hft, wft) + xfft
        out= self.custom_unflatten(out,hft,wft) +xfft

        out=torch.fft.irfft2(out,norm='ortho')
        #out=F.normalize(out)
        out=self.dropout(out)
        return out

#model=spectralModel(16,16,1).to('cuda')
#x=torch.randn(2,16,4,4).to('cuda')
#out=model(x)
#print(out)