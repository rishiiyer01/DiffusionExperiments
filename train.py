#train file with loss fn


#unconditional cifar10 generation

#we start with gaussian noise and denoise in frequency space, we don't let frequencies touch each other, so all linear layers and attention
#are in frequency space


#due to convolution theorem a linear layer in frequency space represents a convolution, but the kernel is only a single frequency

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from model import spectralModel

import numpy as np

#datasets
from torchvision import transforms

# Define the transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor and scale to [0, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Create the dataset with the defined transform
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)


#for labels we have to fft them and the goal is to match the magnitude and phase with mse
def loss_fn(x, labels):
    # Convert labels to frequency domain
    label_fft = torch.fft.rfft2(labels, norm='ortho')
    
    # Flatten both x and label_fft
    x_flat = x.reshape(x.shape[0], -1)
    label_fft_flat = label_fft.reshape(label_fft.shape[0], -1)
    
    # Calculate the magnitude of the complex values
    x_mag = torch.abs(x_flat)
    label_fft_mag = torch.abs(label_fft_flat)
    
    # Calculate the phase of the complex values
    x_phase = torch.angle(x_flat)
    label_fft_phase = torch.angle(label_fft_flat)
    
    # Compute MSE loss for magnitude
    mag_loss = F.mse_loss(x_mag, label_fft_mag)
    
    # Compute circular MSE loss for phase
    phase_diff = torch.remainder(x_phase - label_fft_phase + np.pi, 2 * np.pi) - np.pi
    phase_loss = torch.mean(phase_diff ** 2)
    
    # Combine magnitude and phase losses
    total_loss = mag_loss + phase_loss
    
    return total_loss
    