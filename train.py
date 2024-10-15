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

import numpy as np

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



    
    def __getitem__(self, index):
        image = self.dataset[index][0]  # Get only the image
        image_fft = torch.fft.rfft2(image, norm='ortho')
        
        # Flatten the FFT
        c, h, w = image_fft.shape
        #flattened_fft = image_fft.reshape(b, -1)
        #image_fft=image_fft.unsqueeze(0)
        flattened_fft=custom_flatten(image_fft)
        
        #flattened_fft=flattened_fft.squeeze()
        
        #print(flattened_fft.shape)
        #print(flattened_fft[:, :-1].shape,flattened_fft[:, -1].shape)
        # Shift the FFT data
        target_fft = torch.cat((flattened_fft[:, -1].unsqueeze(1), flattened_fft[:, :-1]), dim=-1)
        #this is so confusing bruh
        #print(target_fft.shape)
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

#for labels we have to fft them and the goal is to match the magnitude and phase with mse
def loss_fn(x, labels):
    # Convert labels to frequency domain
    #label_fft = torch.fft.rfft2(labels, norm='ortho')
    label_fft=labels
    # Flatten both x and label_fft
    x_flat = x.reshape(x.shape[0], -1)
    #x_flat=spectralTransformerBlock.custom_flatten(x)
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
    
model=spectralModel(3,256,16) #in_channels,out_channels,num_blocks

optimizer=optim.Adam(model.parameters(),lr=0.001)



#from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

# Initialize FID metric
#fid = FrechetInceptionDistance(feature=64)

# Training loop
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    # Training loop
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, targets = batch
        images = images.to(device)
        targets=targets.to(device)
        
        optimizer.zero_grad()
        
        # Generate noise
        noise = torch.randn_like(images).to(device)
        
        # Forward pass
        output = model(noise)[1]
        
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