#train file with loss fn


#unconditional cifar10 generation



#due to convolution theorem a linear layer in frequency space represents a convolution, but the kernel is only a single frequency

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from model import SpectralProcessingModel
from functools import reduce
import operator
import numpy as np
import math
from torch.distributions import Categorical
#datasets
from torchvision import transforms
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import wandb
from PIL import Image
from datasets import load_dataset

wandb.init(project="spectral-model", name="training-run")

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

    
def custom_flatten(tensor):
    channel, height, width = tensor.shape
    device = tensor.device
            
    indices, flat_indices, _ = create_index_maps_fftshift(height, width)
    indices = indices.to(device)
    flat_indices = flat_indices.to(device)
            
    flattened = torch.zeros( channel, height * width, device=device).to(torch.cfloat)
    flattened[ :, flat_indices] = tensor[ :, indices[:, 0], indices[:, 1]]
    return flattened
    


def custom_unflatten(flattened, height, width):
    batch, channel, _ = flattened.shape
    device = flattened.device
        
    indices, _, unflatten_indices = create_index_maps_fftshift(height, width)
    indices = indices.to(device)
    unflatten_indices = unflatten_indices.to(device)
        
    unflattened = torch.zeros(batch, channel, height, width, device=device).to(torch.cfloat)
    unflattened[:, :, indices[:, 0], indices[:, 1]] = flattened[:, :, unflatten_indices]
        
    return unflattened
    
class SpectralImageDataset(torch.utils.data.Dataset):
    def __init__(self, split='unlabeled'):
        # Keep existing initialization code
        cifar10_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        self.dataset = cifar10_dataset
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

    def quantize_complex(self, target_fft):
        # target_fft shape is [c, h*w]
        real_part = torch.real(target_fft)  # [c, h*w]
        imag_part = torch.imag(target_fft)  # [c, h*w]
     
        
        

        
        # Create bins for real and imaginary parts
        # Real part range: -7.5 to 25
        real_bins = torch.linspace(-7.5, 25, 1024)
        # Imaginary part range: -7.5 to 7.5
        imag_bins = torch.linspace(-7.5, 7.5, 1024)
        
        # Move bins to target device
        real_bins = real_bins.to(target_fft.device)
        imag_bins = imag_bins.to(target_fft.device)
        
        # Get indices for each value
        real_indices = torch.bucketize(real_part, real_bins)
        imag_indices = torch.bucketize(imag_part, imag_bins)
        
        # Safety clipping
        real_indices = torch.clamp(real_indices, 0, 1023)
        imag_indices = torch.clamp(imag_indices, 0, 1023)
        
        # Create one-hot encodings
        real_one_hot = torch.zeros(3, 1024, target_fft.shape[1]).to(target_fft.device)
        imag_one_hot = torch.zeros(3, 1024, target_fft.shape[1]).to(target_fft.device)
        
        # Fill one-hot encodings
        for c in range(3):
            real_one_hot[c, :, :] = F.one_hot(real_indices[c], num_classes=1024).float().t()
            imag_one_hot[c, :, :] = F.one_hot(imag_indices[c], num_classes=1024).float().t()

        return real_one_hot, imag_one_hot
        
    def __getitem__(self, index):
        image_data = self.dataset[index][0]
        condition=self.dataset[index][1]
        
        image = self.transform(image_data)
        
        # Process image through FFT
        image_fft = torch.fft.rfft2(image, norm='ortho')
        image_fft = torch.fft.fftshift(image_fft, dim=1)
        
        # Flatten and quantize
        flattened_fft = custom_flatten(image_fft)
        target_fft = self.quantize_complex(flattened_fft)
        
        return image, target_fft,condition

    def __len__(self):
        return len(self.dataset)

# Create dataset and dataloader
dataset = SpectralImageDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=10, pin_memory=True,pin_memory_device="cuda",persistent_workers=True)





    
#for labels we have to fft them and the goal is to match the magnitude and phase class distributions with cross entropy


lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze',normalize=True).to('cuda')


def loss_fn(x, labels):
    x_real = x[0]    # [b,c,256,x*y]
    x_imag = x[1]    # [b,c,256,x*y]
    target_real = labels[0]    # [b,c,256,x*y]
    target_imag = labels[1]    # [b,c,256,x*y]
    
    b, c, vocab, xy = x_real.shape
    
    # Reshape predictions and targets
    x_real = x_real.permute(0, 1, 3, 2).reshape(-1, vocab)      # [b*c*xy, 256]
    x_imag = x_imag.permute(0, 1, 3, 2).reshape(-1, vocab)      # [b*c*xy, 256]
    target_real = target_real.permute(0, 1, 3, 2).reshape(-1, vocab)    # [b*c*xy, 256]
    target_imag = target_imag.permute(0, 1, 3, 2).reshape(-1, vocab)    # [b*c*xy, 256]

    # Get target indices for cross entropy loss
    target_real_indices = torch.argmax(target_real, dim=-1)    # [b*c*xy]
    target_imag_indices = torch.argmax(target_imag, dim=-1)    # [b*c*xy]
    
    # Classification losses
   
    
    real_loss = F.cross_entropy(x_real, target_real_indices)
    imag_loss = F.cross_entropy(x_imag, target_imag_indices)
    
    # Setup for reconstruction
    real_dist = torch.linspace(-7.5, 25, 1024).to(x_real.device)
    imag_dist = torch.linspace(-7.5, 7.5, 1024).to(x_imag.device)

    # Get predicted indices
    real_indices = torch.argmax(x_real, dim=-1)
    imag_indices = torch.argmax(x_imag, dim=-1)
    
    # Get continuous values
    real_values = real_dist[real_indices]
    imag_values = imag_dist[imag_indices]
    
    # Target values
    target_real_value = real_dist[target_real_indices]
    target_imag_value = imag_dist[target_imag_indices]
    
    # Reconstruction for target
    image_target = target_real_value + 1j * target_imag_value
    
    
    image_target = image_target.reshape(b, c, xy)
    image_target = custom_unflatten(image_target, 32, 17)
    image_target = torch.fft.ifftshift(image_target, dim=2)
    image_target = torch.fft.irfft2(image_target, norm='ortho')
    
    # Reconstruction for prediction
    out_ft = real_values + 1j * imag_values 
    
    out_ft = out_ft.reshape(b, c, xy)
    x_img = custom_unflatten(out_ft, 32, 17)
    x_img = torch.fft.ifftshift(x_img, dim=2)
    x_img = torch.fft.irfft2(x_img, norm='ortho')
    
    # Normalize images
    x_img = (x_img - torch.min(x_img)) / (torch.max(x_img) - torch.min(x_img))
    image_target = (image_target - torch.min(image_target)) / (torch.max(image_target) - torch.min(image_target))
    
    # LPIPS loss
    lpips_loss = lpips(x_img, image_target)
    
    # MSE losses on continuous values
    real_mse = F.mse_loss(real_values, target_real_value)
    imag_mse = F.mse_loss(imag_values, target_imag_value)
    
    # Total loss
    total_loss = real_loss/math.log(1024) + imag_loss/math.log(1024) #+ lpips_loss
    
    return total_loss, real_loss, imag_loss, real_mse, imag_mse, lpips_loss

num_epochs=1
model = SpectralProcessingModel(3, 512, 24)
model.load_state_dict(torch.load('spectral_model10.pth'))
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
max_grad_norm = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

print('param count', count_params(model))
from tqdm import tqdm
for epoch in range(num_epochs):
    model.train()
    total = 0
    
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, targets,condition = batch
        images = images.to(device)
        targets[0] = targets[0].to(device)
        targets[1] = targets[1].to(device)
        condition=condition.to(device).to(torch.float)
        
        optimizer.zero_grad()
        output = model(images,condition)
        
        total_loss, real_loss, imag_loss, real_mse, imag_mse, lpips_loss = loss_fn(output, targets)
        
        if not torch.isnan(total_loss):
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total += total_loss.item()
            
            wandb.log({
                'total_loss': total_loss.item(),
                'real_loss': real_loss.item(),
                'imag_loss': imag_loss.item(),
                'real_mse': real_mse.item(),
                'imag_mse': imag_mse.item(),
                'lpips_loss': lpips_loss.item(),
            })
    
    avg_loss = total / len(dataloader)
    wandb.log({'avg_loss': avg_loss})
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), 'spectral_model10.pth')
print("Training completed and model saved.")