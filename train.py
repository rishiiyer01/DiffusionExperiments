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

from torchvision import transforms
class ImageOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index][0]  # Return only the image

    def __len__(self):
        return len(self.dataset)

# Define the transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create the dataset with the defined transform
cifar10_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataset = ImageOnlyDataset(cifar10_dataset)

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)


#for labels we have to fft them and the goal is to match the magnitude and phase with mse
def loss_fn(x, labels,step,total_steps):
    # Convert labels to frequency domain
    label_fft = torch.fft.rfft2(labels, norm='ortho')
    b,c,h,w=label_fft.shape
    numfreqs=step*w//total_steps #w is n/2
    
    label_fft_subset1=label_fft[:,:,:numfreqs,:numfreqs]
    label_fft_subset2=label_fft[:,:,-numfreqs:,:numfreqs]
    label_fft_subset=torch.cat((label_fft_subset1,label_fft_subset2),dim=2)
    x_subset1=x[:,:,:numfreqs,:numfreqs]
    x_subset2=x[:,:,-numfreqs:,:numfreqs]
    x_subset=torch.cat((x_subset1,x_subset2),dim=2)
    # Flatten both x and label_fft
    x_flat = x_subset.reshape(x.shape[0], -1)
    label_fft_flat = label_fft_subset.reshape(label_fft.shape[0], -1)
    
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
total_steps=16
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    # Training loop
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = batch
        images = images.to(device)
        
        optimizer.zero_grad()
        
        # Generate noise
        noise = torch.randn_like(images).to(device)
        
        for step in range(total_steps):
            output, fft = model(noise) #potentially will add timestep embedding later after initial experiments
             #step/total steps also represents fraction of image frequencies
            loss = loss_fn(fft,images,step,total_steps)
            loss.backward()
            total_loss += loss.item()
            noise=output.detach()
        
        
        
        optimizer.step()
        
        
    
    avg_loss = total_loss / len(dataloader)/total_steps
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    

# Save the trained model
torch.save(model.state_dict(), 'spectral_model.pth')
print("Training completed and model saved.")
    