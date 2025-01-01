import torch
import torch.nn as nn
import torch.nn.functional as F
from math import inf
import math
import sys
sys.path.append('COSMOS')
from cosmos_tokenizer.image_lib import ImageTokenizer


class cosmos_vae(nn.Module):
    def __init__(self):
        super().__init__()
        model_name = "Cosmos-Tokenizer-CI16x16"
        self.encoder = ImageTokenizer(checkpoint_enc=f'pretrained_ckpts/{model_name}/encoder.jit')
        self.decoder=ImageTokenizer(checkpoint_dec=f'pretrained_ckpts/{model_name}/decoder.jit')

#no need for forward method, the encoder and decoder have forward methods


def create_index_maps_fftshift(height, width):
    """
    Creates index mappings for fft-shifted data starting from center-left.
    Returns both flattening and unflattening indices.
    """
    indices = []
    flat_indices = []
    index = 0
    
    # Find center row
    center_h = height // 2
    
    # Start from center row, then alternate up and down
    row_order = [center_h]
    for i in range(1, height):
        if center_h + i < height:
            row_order.append(center_h + i)
        if center_h - i >= 0:
            row_order.append(center_h - i)
    
    # Process each row from left to right
    for h in row_order:
        for w in range(width):
            indices.append((h, w))
            flat_indices.append(index)
            index += 1
    
    indices = torch.tensor(indices)
    flat_indices = torch.tensor(flat_indices)
    
    # Create inverse mapping for unflattening
    unflatten_indices = torch.zeros_like(flat_indices)
    unflatten_indices[flat_indices] = torch.arange(len(flat_indices))
    
    return indices, flat_indices, unflatten_indices



class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096):
        """
        Initialize Rotary Position Embedding
        
        Args:
            dim: Dimension of the embedding (must be divisible by 2)
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        if dim % 2 != 0:
            raise ValueError("Dimension must be divisible by 2")
            
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Create position indices tensor
        position = torch.arange(max_seq_len).unsqueeze(1)  # [max_seq_len, 1]
        
        # Create dimension indices tensor for half the dimension
        # Since we'll rotate half the dimensions, we only need dim/2
        div_term = torch.exp(
            torch.arange(0, dim//2) * -(math.log(10000.0) / (dim//2))
        )
        
        # Compute sin and cos tables for half dimensions
        emb = position * div_term
        self.register_buffer("sin_table", emb.sin().unsqueeze(0))  # [1, max_seq_len, dim//2]
        self.register_buffer("cos_table", emb.cos().unsqueeze(0))  # [1, max_seq_len, dim//2]
        
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embedding to input tensor
        
        Args:
            x: Input tensor of shape [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            Tensor with positional information encoded
        """
        batch_size, num_heads, seq_len, dim = x.shape
        
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_seq_len}")
            
        # Get sin and cos values for current sequence length
        sin = self.sin_table[:, :seq_len, :]  # [1, seq_len, dim//2]
        cos = self.cos_table[:, :seq_len, :]  # [1, seq_len, dim//2]
        
        # Duplicate the sin/cos for the full dimension
        sin = torch.cat([sin, sin], dim=-1)  # [1, seq_len, dim]
        cos = torch.cat([cos, cos], dim=-1)  # [1, seq_len, dim]
        
        # Reshape sin and cos for broadcasting
        sin = sin.unsqueeze(1)  # [1, 1, seq_len, dim]
        cos = cos.unsqueeze(1)  # [1, 1, seq_len, dim]
        
        # Expand to match input shape
        sin = sin.expand(batch_size, num_heads, -1, -1)
        cos = cos.expand(batch_size, num_heads, -1, -1)
        
        # Apply rotation using complex number multiplication:
        # (a + ib)(cos θ + i sin θ) = (a cos θ - b sin θ) + i(a sin θ + b cos θ)
        return (x * cos) + (self._rotate_half(x) * sin)
    

class SpectralProcessingModel(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_blocks):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        
        # Initial processing
        self.initial_proj = nn.Linear(in_channels * 2, hidden_dim,bias=True)  # *2 for magnitude and phase
        
        # Main processing blocks
        self.blocks = nn.ModuleList([
            SpectralBlock(hidden_dim) for _ in range(num_blocks)
        ])
        vae=cosmos_vae()
        self.encoder=vae.encoder.to('cuda')
        # Output projections for magnitude and phase
        self.mag_proj = nn.Linear(hidden_dim//2, 1024*in_channels,bias=True)  # 1024, used to be 256 classes for magnitude
        self.phase_proj = nn.Linear(hidden_dim//2, 1024*in_channels,bias=True)  # 1024, used to be 256 classes for phase
        self.dc_linear=nn.Linear(1,hidden_dim,bias=True)
        #self.start_token = nn.Parameter(torch.randn(1, 1, 1))
        
        
    def custom_flatten(self, tensor):
        """
        Flattens fft-shifted tensor starting from center frequencies
        """
        batch, channel, height, width = tensor.shape
        device = tensor.device
        
        indices, flat_indices, _ = create_index_maps_fftshift(height, width)
        indices = indices.to(device)
        flat_indices = flat_indices.to(device)
        
        flattened = torch.zeros(batch, channel, height * width, device=device).to(tensor.dtype)
        flattened[:, :, flat_indices] = tensor[:, :, indices[:, 0], indices[:, 1]]
        
        return flattened
    
    def custom_unflatten(self, tensor, height, width):
        """
        Unflattens tensor back to original fft-shifted shape
        """
        batch, channel, seq_len = tensor.shape
        device = tensor.device
        
        indices, _, unflatten_indices = create_index_maps_fftshift(height, width)
        indices = indices.to(device)
        unflatten_indices = unflatten_indices.to(device)
        
        unflattened = torch.zeros(batch, channel, height, width, device=device).to(tensor.dtype)
        unflattened[:, :, indices[:, 0], indices[:, 1]] = tensor[:, :, unflatten_indices]
        
        return unflattened


        
    def forward(self, x,condition):
        #encode image into patches
        x=x.to(torch.bfloat16)
        with torch.no_grad():
            
            (x,)=self.encoder.encode(x) #b,c,x,y  
            x=x.to(torch.float)
        # Initial FFT
        x_ft = torch.fft.rfft2(x, norm='ortho')
        x_ft=torch.fft.fftshift(x_ft,dim=2)
        # Convert to magnitude and phase, this isn't actually magnitude and phase I just didn't change the code yet
        magnitude = torch.real(x_ft)
         
        
        #magnitude=torch.abs(x_ft)
        phase = torch.imag(x_ft)
        
        
        # Flatten and concatenate
        b, c, h, w = magnitude.shape
        phase=self.custom_flatten(phase)

        magnitude=self.custom_flatten(magnitude) #returns shape of b,c,h*w
        
        x = torch.stack([magnitude, phase], dim=1) #returns shape of b,2,c,h*w
        x=x.permute(0,3,1,2)
        x=x.reshape(b,h*w,c*2) #reshape is invertible so we can recover channel, then we can recover abs, phase
        
        
        # Initial projection
        x = self.initial_proj(x)
        
        #the condition is the start token
        condition=condition.unsqueeze(1).unsqueeze(1)
        first_freq=self.dc_linear(condition) #b,1,hidden_dim
        #cat dc freq to image
        x=torch.cat((first_freq,x[:,:-1,:]),dim=1)
        
        for block in self.blocks:
            x = block(x)

        x=x.reshape(b,h*w,2,-1)
        # Project to classes
        mag_logits = self.mag_proj(x)[:,:,0,:]
        phase_logits = self.phase_proj(x)[:,:,1,:] #shape of b,h*w,256*3
        
        # Reshape to match your current output format
        mag_logits = mag_logits.view(b, h*w, 1024, c).permute(0,3,2,1)
        phase_logits = phase_logits.view(b, h*w, 1024, c).permute(0,3,2,1) #vocab size used to be 256
        
        
        return mag_logits, phase_logits




class SpectralBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention =SpectralAttention(hidden_dim)
        self.ff = FeedForward(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


        
#checked and is correct
class FeedForward(nn.Module):
    def __init__(self, hidden_dim, expansion_factor=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim)
        )
        
    def forward(self, x):
        return self.net(x)

#classic style attention
class SpectralAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.rope=RotaryPositionEmbedding(self.head_dim)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        B, N,dim = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        #print(q.shape)
        q=q+self.rope(q)
        k=k+self.rope(k)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask.to(attn.device), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        
        # Output projection
        x = self.proj(x)
        return x
