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
from model import spectralModel, spectralTransformerBlock
from functools import reduce
import operator
import numpy as np
import math
#datasets
from torchvision import transforms


def create_index_maps(height, width):
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
    
def custom_flatten(tensor):
    channel, height, width = tensor.shape
    device = tensor.device
            
    indices, flat_indices = create_index_maps(height, width)
    indices = indices.to(device)
    flat_indices = flat_indices.to(device)
            
    flattened = torch.zeros( channel, height * width, device=device).to(torch.cfloat)
    flattened[ :, flat_indices] = tensor[ :, indices[:, 0], indices[:, 1]]
    return flattened
    
class SpectralImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset




    def quantize(self, target_fft):
        # target_fft shape is [c, h*w]
        target_fft_phase = torch.angle(target_fft)  # [c, h*w]
        target_fft_abs = torch.abs(target_fft)      # [c, h*w]
        
        # Create bins - directly using 64 points
        phase_bins = torch.linspace(-math.pi, math.pi, 64).to(target_fft.device)  
        abs_bins = torch.linspace(0, 0.7, 64).to(target_fft.device)              
        
        # Get indices for each value
        phase_indices = torch.bucketize(target_fft_phase, phase_bins)  # [c, h*w]
        abs_indices = torch.bucketize(target_fft_abs, abs_bins)        # [c, h*w]
        
        # Create one-hot encodings
        phase_one_hot = torch.zeros(3, 64, target_fft.shape[1]).to(target_fft.device)  # [c, 64, h*w]
        abs_one_hot = torch.zeros(3, 64, target_fft.shape[1]).to(target_fft.device)    # [c, 64, h*w]
        
        # Fill one-hot encodings
        for c in range(3):  # For each channel
            phase_one_hot[c] = F.one_hot(phase_indices[c], num_classes=64).float().t()
            abs_one_hot[c] = F.one_hot(abs_indices[c], num_classes=64).float().t()

        abs_one_hot=abs_one_hot.permute(1,0,2)
        phase_one_hot=phase_one_hot.permute(1,0,2)
        return abs_one_hot, phase_one_hot  # Returns tensors of shape [64,c, h*w]
        
    def __getitem__(self, index):
        image = self.dataset[index][0]  # Get only the image
        #image=image/torch.linalg.norm(image)
        image=F.normalize(image)
        image_fft = torch.fft.rfft2(image, norm='ortho')
        #print(image_fft[:,0,0])
        #image_fft=image_fft/torch.linalg.norm(image_fft)
        #print(torch.linalg.norm(image_fft))
        # Flatten the FFT
        c, h, w = image_fft.shape
        #flattened_fft = image_fft.reshape(b, -1)
        #image_fft=image_fft.unsqueeze(0)
        flattened_fft=custom_flatten(image_fft)
        
        #flattened_fft=flattened_fft.squeeze()
        
        #print(flattened_fft.shape)
        #print(flattened_fft[:, :-1].shape,flattened_fft[:, -1].shape)
        # Shift the FFT data
        #target_fft = torch.cat((flattened_fft[:, -1].unsqueeze(1), flattened_fft[:, :-1]), dim=-1)
        target_fft=torch.roll(flattened_fft,shifts=-1,dims=1)
        target_fft=target_fft/torch.linalg.norm(target_fft)
        target_fft=self.quantize(target_fft)
        
        
        return image, target_fft

    def __len__(self):
        return len(self.dataset)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
cifar10_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataset = SpectralImageDataset(cifar10_dataset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)





    
#for labels we have to fft them and the goal is to match the magnitude and phase class distributions with cross entropy
def loss_fn(x, labels):
    # x.shape=b,vocab,c,x*y
    x_abs = x[0]    # [b,64,c,x*y]
    x_phase = x[1]  # [b,64,c,x*y]
    
    target_abs_fft = labels[0]    # [b,64,c,x*y]
    target_phase_fft = labels[1]  # [b,64,c,x*y]
    
    # Reshape for cross entropy
    # Need to reshape to [B*C*XY, 64] for predictions and [B*C*XY] for targets
    b, vocab, c, xy = x_abs.shape
    
    # Predictions: reshape and transpose to get vocab dimension last
    x_abs = x_abs.permute(0, 2, 3, 1).reshape(-1, vocab)      # [b*c*xy, 64]
    x_phase = x_phase.permute(0, 2, 3, 1).reshape(-1, vocab)  # [b*c*xy, 64]
    
    # Targets: get indices from one-hot
    target_abs = target_abs_fft.permute(0, 2, 3, 1).reshape(-1, vocab)    # [b*c*xy, 64]
    target_phase = target_phase_fft.permute(0, 2, 3, 1).reshape(-1, vocab)  # [b*c*xy, 64]
    
    # Convert one-hot targets to class indices
    target_abs = torch.argmax(target_abs, dim=-1)    # [b*c*xy]
    target_phase = torch.argmax(target_phase, dim=-1)  # [b*c*xy]
    
    # Compute cross entropy losses
    mag_loss = F.cross_entropy(x_abs, target_abs)
    phase_loss = F.cross_entropy(x_phase, target_phase)
    
    # Combine magnitude and phase losses
    total_loss = mag_loss + phase_loss
    
    return total_loss
    
model=spectralModel(3,512,24) #in_channels,out_channels,num_blocks

optimizer=optim.Adam(model.parameters(),lr=0.001)



#from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

# Initialize FID metric
#fid = FrechetInceptionDistance(feature=64)

# Training loop
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

print('param count',count_params(model))
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    # Training loop
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, targets = batch
        images = images.to(device)
        images=images/torch.linalg.norm(images)
        targets[0]=targets[0].to(device)
        targets[1]=targets[1].to(device)
        
        optimizer.zero_grad()
        
        # Generate noise
        #noise = torch.randn_like(images).to(device)
        
        # Forward pass
        output = model(images)
        
        # Compute loss
        loss = loss_fn(output, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    

# Save the trained model
torch.save(model.state_dict(), 'spectral_model.pth')
print("Training completed and model saved.")