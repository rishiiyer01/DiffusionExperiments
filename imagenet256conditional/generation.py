import torch
import torch.nn as nn
import torch.nn.functional as F
from math import inf
import math
from torch.distributions import Categorical

from model import SpectralProcessingModel

from model import SpectralProcessingModel, custom_unflatten

class SpectralGenerationModel(SpectralProcessingModel):
    def __init__(self, in_channels, hidden_dim, num_blocks):
        super().__init__(in_channels, hidden_dim, num_blocks)
        
    def topK(self, logits, k, temperature=1.0):
        zeros = logits.new_ones(logits.shape) * float('-inf')
        values, indices = torch.topk(logits, k, dim=-1)
        zeros.scatter_(-1, indices, values)
        dist = Categorical(logits=zeros / temperature)
        return dist.sample()
        
    def forward(self, condition, batch_size=1, temperature=1.0):
        device = next(self.parameters()).device
        
        # Process condition (class label) same as training
        condition = condition.to(torch.float).to(device)
        
        # Start with DC component
        x = self.dc_linear(condition.unsqueeze(1).unsqueeze(1))  # Match training dimensions
        
        # Initialize tensors for storing outputs
        h, w = 16, 9  # latent dimensions after FFT
        seq_len = h * w
        channels = 16  # number of channels from encoder
        
        real_outputs = torch.zeros(batch_size, channels, 1024, seq_len).to(device)
        imag_outputs = torch.zeros(batch_size, channels, 1024, seq_len).to(device)
        storage_real = torch.zeros(batch_size, channels, seq_len).to(device)
        storage_imag = torch.zeros(batch_size, channels, seq_len).to(device)
        
        # Generate one frequency at a time
        for i in range(seq_len):
            # Forward pass through transformer blocks
            curr_x = x
            for block in self.blocks:
                curr_x = block(curr_x)
            
            curr_x = curr_x.reshape(batch_size, i+1, 2, -1)
            curr_real = self.mag_proj(curr_x[:,-1:,0,:])
            curr_imag = self.phase_proj(curr_x[:,-1:,1,:])
            
            # Match training dimensions
            real_outputs[:, :, :, i] = curr_real.view(batch_size, channels, 1024)
            imag_outputs[:, :, :, i] = curr_imag.view(batch_size, channels, 1024)
            
            if i < seq_len - 1:
                # Adaptive sampling based on frequency position
                k = max(2, 5 - (2 * i // (seq_len//2)))
                real_sample = self.topK(curr_real.view(batch_size, channels, 1024), k, temperature)
                imag_sample = self.topK(curr_imag.view(batch_size, channels, 1024), k, temperature)
                
                # Get continuous values from discrete bins (matching training ranges)
                real_dist = torch.linspace(-7, 7, 1024).to(device)
                imag_dist = torch.linspace(-7, 7, 1024).to(device)
                
                real_values = real_dist[real_sample]
                imag_values = imag_dist[imag_sample]
                
                storage_real[:,:,i] = real_values
                storage_imag[:,:,i] = imag_values
                
                # Prepare next input (matching training format)
                next_freq = torch.stack([real_values, imag_values], dim=1)
                next_freq = next_freq.reshape(batch_size, 1, -1)
                next_freq = self.initial_proj(next_freq)
                
                x = torch.cat([x, next_freq], dim=1)
        
        # Return both discretized outputs and continuous values for reconstruction
        return real_outputs, imag_outputs, storage_real, storage_imag
    
    def reconstruct_from_samples(self, real_values, imag_values):
        """
        Reconstruct image from generated spectral components
        """
        batch_size = real_values.shape[0]
        
        # Convert to complex numbers
        out_ft = torch.sign(real_values) * (torch.exp(torch.abs(real_values))-1) + 1j * imag_values
        
        # Reshape and unflatten
        x_img = custom_unflatten(out_ft, 16, 9)
        x_img = torch.fft.ifftshift(x_img, dim=2)
        x_img = torch.fft.irfft2(x_img, norm='ortho')
        
        # Decode through VAE decoder
        with torch.no_grad():
            x_img = x_img.to(torch.bfloat16)
            x_img = self.decoder.decode(x_img)
            x_img = x_img.to(torch.float)
        
        return x_img