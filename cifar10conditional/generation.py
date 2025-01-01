import torch
import torch.nn as nn
import torch.nn.functional as F
from math import inf
import math
from torch.distributions import Categorical

from model import SpectralProcessingModel



class SpectralGenerationModel(SpectralProcessingModel):
    def __init__(self, in_channels, hidden_dim, num_blocks):
        super().__init__(in_channels, hidden_dim, num_blocks)
        
    def topK(self, logits, k, temperature=1.0):
        zeros = logits.new_ones(logits.shape) * float('-inf')
        values, indices = torch.topk(logits, k, dim=-1)
        zeros.scatter_(-1, indices, values)
        dist = Categorical(logits=zeros / temperature)
        return dist.sample()
        
    def forward(self,condition, batch_size=1, temperature=1.0):
        device = next(self.parameters()).device
        
        # Start with condition
        #first_freq = self.start_token.expand(batch_size, 1, 1)
        condition=torch.tensor(condition).to(torch.float).to(device)
        condition=condition.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = self.dc_linear(condition)  # [b, 1, hidden_dim]
        print(x.shape)
        
        # Initialize tensors for storing outputs
        h, w = 32, 17  # CIFAR10 after FFT
        seq_len = h * w
        
        real_outputs = torch.zeros(batch_size, 3, 1024, seq_len).to(device) 
        imag_outputs = torch.zeros(batch_size, 3, 1024, seq_len).to(device)
        storage_real = torch.zeros(batch_size, 3, seq_len).to(device)
        storage_imag = torch.zeros(batch_size, 3, seq_len).to(device)
        
        # Generate one frequency at a time
        for i in range(seq_len):
            # Forward pass through transformer blocks
            curr_x = x.clone()
            for block in self.blocks:
                curr_x = block(curr_x)
                
            curr_x = curr_x.reshape(batch_size, i+1, 2, -1)
            curr_real = self.mag_proj(curr_x[:,-1:,0,:])  # Changed from mag_proj
            curr_imag = self.phase_proj(curr_x[:,-1:,1,:])  # Changed from phase_proj
            
            # Reshape to [b, 3, 256]
            curr_real = curr_real.view(batch_size, 1024, 3).permute(0, 2, 1)
            curr_imag = curr_imag.view(batch_size, 1024, 3).permute(0, 2, 1)
            
            real_outputs[:, :, :, i] = curr_real
            imag_outputs[:, :, :, i] = curr_imag
            
            if i < seq_len - 1:
                if i < seq_len//4:
                    real_sample = self.topK(curr_real, 5, temperature=1.0)
                    imag_sample = self.topK(curr_imag, 5, temperature=1.0)
                
                elif i < seq_len//2:
                    real_sample = self.topK(curr_real,3, temperature=1.0)
                    imag_sample = self.topK(curr_imag, 3, temperature=1.0)

                else:
                    real_sample = self.topK(curr_real, 2, temperature=1.0)
                    imag_sample = self.topK(curr_imag, 2, temperature=1.0)
                
                # Get continuous values from discrete bins
                #real_dist = torch.linspace(-7.5, 25, 1024).to(device)
                real_dist=torch.linspace(-7.5,25,1024).to(device)
                imag_dist = torch.linspace(-7.5, 7.5, 1024).to(device)
                
                real_values = real_dist[real_sample]  # [b, 3]
                imag_values = imag_dist[imag_sample]  # [b, 3]
                
                storage_real[:,:,i] = real_values
                storage_imag[:,:,i] = imag_values
                
                # Stack and prepare next input
                next_freq = torch.stack([real_values, imag_values], dim=1)  # [b, 2, 3]
                next_freq = next_freq.reshape(batch_size, 1, -1)  # [b, 1, 6]
                next_freq = self.initial_proj(next_freq)  # [b, 1, hidden_dim]
                
                x = torch.cat([x, next_freq], dim=1)
                
        
        return real_outputs, imag_outputs, storage_real, storage_imag