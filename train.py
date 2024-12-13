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
        # Load the dataset
        #self.dataset = load_dataset("jxie/stl10")[split]
        cifar10_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        self.dataset=cifar10_dataset
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Tiny ImageNet images are 64x64, cifar 10 is 32x32
            transforms.ToTensor(),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet stats
        ])
        #max_samples=50000
        #self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

    def quantize(self, target_fft):
        # target_fft shape is [c, h*w]
        target_fft_phase = torch.angle(target_fft)  # [c, h*w]
        target_fft_abs = torch.log1p(torch.abs(target_fft))  # [c, h*w]
        
        # Create bins
        phase_bins = torch.linspace(-math.pi, math.pi, 256)  
        abs_bins = torch.linspace(0, 3.5, 256)              
        
        # Move bins to target device
        phase_bins = phase_bins.to(target_fft.device)
        abs_bins = abs_bins.to(target_fft.device)
        
        # Get indices for each value
        phase_indices = torch.bucketize(target_fft_phase, phase_bins)  # [c, h*w]
        abs_indices = torch.bucketize(target_fft_abs, abs_bins)        # [c, h*w]
        
        # Safety clipping
        phase_indices = torch.clamp(phase_indices, 0, 255)
        abs_indices = torch.clamp(abs_indices, 0, 255)
        
        # Create one-hot encodings
        phase_one_hot = torch.zeros(3, 256, target_fft.shape[1]).to(target_fft.device)
        abs_one_hot = torch.zeros(3, 256, target_fft.shape[1]).to(target_fft.device)
        
        # Fill one-hot encodings
        for c in range(3):
            phase_one_hot[c, :, :] = F.one_hot(phase_indices[c], num_classes=256).float().t()
            abs_one_hot[c, :, :] = F.one_hot(abs_indices[c], num_classes=256).float().t()

        return abs_one_hot, phase_one_hot
        
    def __getitem__(self, index):
        # Get image from dataset
        image_data = self.dataset[index][0]
        
        # Convert to PIL Image if necessary
        #if not isinstance(image_data, Image.Image):
            #image_data = Image.fromarray(image_data)
        
        # Apply transforms
        image = self.transform(image_data)
        
        # Process image through FFT
        image_fft = torch.fft.rfft2(image, norm='ortho')
        image_fft = torch.fft.fftshift(image_fft, dim=1)
        
        # Flatten and quantize
        flattened_fft = custom_flatten(image_fft)
        target_fft = self.quantize(flattened_fft)
        
        return image, target_fft

    def __len__(self):
        return len(self.dataset)

# Create dataset and dataloader
dataset = SpectralImageDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=10, pin_memory=True,pin_memory_device="cuda",persistent_workers=True)





    
#for labels we have to fft them and the goal is to match the magnitude and phase class distributions with cross entropy


lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze',normalize=True).to('cuda')


def loss_fn(x, labels):
    # x.shape=b,vocab,c,x*y
    x_abs = x[0]    # [b,c,256,x*y]
    x_phase = x[1]  # [b,c,256,x*y]
    #print(x_abs.shape)
    target_abs_fft = labels[0]    # [b,c,256,x*y]
    target_phase_fft = labels[1]  # [b,c,256,x*y]
    
    # Reshape for cross entropy
    # Need to reshape to [B*C*XY, 64] for predictions and [B*C*XY] for targets
    b, c, vocab, xy = x_abs.shape
    

    
    # Predictions: reshape and transpose to get vocab dimension last
    x_abs = x_abs.permute(0, 1, 3, 2).reshape(-1, vocab)      # [b*c*xy, 256]
    x_phase = x_phase.permute(0, 1, 3, 2).reshape(-1, vocab)  # [b*c*xy, 256]
    
    # Targets: get indices from one-hot
    target_abs = target_abs_fft.permute(0, 1, 3, 2).reshape(-1, vocab)    # [b*c*xy, 256]
    target_phase = target_phase_fft.permute(0, 1, 3, 2).reshape(-1, vocab)  # [b*c*xy, 256]

    
    
    # Convert one-hot targets to class indices
    #print(target_abs[0,:])
    target_abs = torch.argmax(target_abs, dim=-1)    # [b*c*xy]
    target_phase = torch.argmax(target_phase, dim=-1)  # [b*c*xy]
    #print(torch.mean(target_abs.to(torch.float)))
    # Compute cross entropy losses
    #weights for high frequencies
    #weights = torch.exp(-0.05*(torch.linspace(0, 3.5, 256)-3.5))
    mag_loss = F.cross_entropy(x_abs, target_abs)
    phase_loss = F.cross_entropy(x_phase, target_phase)


    
    #getting original image for simple reconstruction loss, might switch to lpips eventually
    #target image
    abs_dist=torch.linspace(0, 3.5, 256).to('cuda')
    phase_dist=torch.linspace(-math.pi, math.pi,256).to('cuda') 
    
    target_abs_value=abs_dist[target_abs]
    target_phase_value=phase_dist[target_phase]
    image_target= (torch.exp(target_abs_value)-1) * torch.exp(1j * target_phase_value)
    image_target=image_target.reshape(b,c,xy)
    image_target=custom_unflatten(image_target,32,17)
    
    image_target=torch.fft.ifftshift(image_target,dim=2)
    image_target=torch.fft.irfft2(image_target, norm='ortho')
    
    #inference image
    temperature = 1
    # Sample with temperature scaling
    
    #absprobdist = Categorical(logits=x_abs / temperature)
    #phaseprobdist = Categorical(logits=x_phase / temperature)
    #abs_sampled = absprobdist.sample()
    #phase_sampled = phaseprobdist.sample()
    abs_sampled=torch.argmax(F.softmax(x_abs,dim=-1),dim=-1)
    phase_sampled=torch.argmax(F.softmax(x_phase,dim=-1),dim=-1)
    
    # Convert indices to values
    abs_sampled = abs_dist[abs_sampled]
    phase_sampled = phase_dist[phase_sampled]

    #phase_mse=F.mse_loss(phase_sampled,target_phase_value)
    phase_diff = torch.remainder(phase_sampled - target_phase_value + np.pi, 2 * np.pi) - np.pi
    phase_mse = torch.mean(phase_diff ** 2)
    abs_mse=F.mse_loss(abs_sampled,target_abs_value)
    # Ensure valid ranges
    #abs_sampled = torch.clamp(abs_sampled, 0, 1)
    #phase_sampled = torch.clamp(phase_sampled, -math.pi, math.pi)
    # Convert to complex
    out_ft = (torch.exp(abs_sampled)-1) * torch.exp(1j * phase_sampled)  # More numerically stable
    out_ft=out_ft.reshape(b,c,xy)
    x_img=custom_unflatten(out_ft,32,17)
    x_img=torch.fft.ifftshift(x_img,dim=2)
    x_img=torch.fft.irfft2(x_img,norm='ortho')


    #x_img=x_img/torch.linalg.norm(x_img)
    #image_target=image_target/torch.linalg.norm(image_target)



    x_img=(x_img-torch.min(x_img))/(torch.max(x_img)-torch.min(x_img))
    image_target=(image_target-torch.min(image_target))/(torch.max(image_target)-torch.min(image_target))
    #x_img = 2 * (x_img-torch.min(x_img))/(torch.max(x_img)-torch.min(x_img)) - 1
    #image_target = 2 * (image_target-torch.min(image_target))/(torch.max(image_target)-torch.min(image_target)) - 1
    #mse_loss=F.mse_loss(x_img,image_target) #reconstruction
    lpipsloss=lpips(x_img,image_target)
    

    
    # Combine magnitude and phase losses, mse reconstruction with 0.3
    total_loss = mag_loss/math.log(256)  +  lpipsloss  + phase_loss/math.log(256)   #+ 0.1*abs_mse+0.1*phase_mse
    
    
    return total_loss, mag_loss, phase_loss, phase_mse,abs_mse, lpipsloss
    
model=SpectralProcessingModel(3,512,24) #in chan hidden dim num blocks
#model.load_state_dict(torch.load('spectral_model10.pth'))
optimizer=optim.Adam(model.parameters(),lr=0.0005)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
max_grad_norm = 1.0


#from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

# Initialize FID metric
#fid = FrechetInceptionDistance(feature=64)

# Training loop
num_epochs = 3
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
    total = 0
    
    # Training loop
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, targets = batch
        images = images.to(device)
        #images=images/torch.linalg.norm(images)
        targets[0]=targets[0].to(device)
        targets[1]=targets[1].to(device)
        
        optimizer.zero_grad()
        
        # Generate noise
        #noise = torch.randn_like(images).to(device)
        
        # Forward pass
        output = model(images)
        
        # Compute loss
        total_loss, mag_loss, phase_loss, phase_mse, abs_mse, lpips_loss = loss_fn(output, targets)
        if not torch.isnan(total_loss):
            total_loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total += total_loss.item()
            # Log metrics to wandb
            wandb.log({
                'total_loss': total_loss.item(),
                'magnitude_loss': mag_loss.item(),
                'phase_loss': phase_loss.item(),
                'phase_mse': phase_mse.item(),
                'lpips_loss': lpips_loss.item(),
                'abs_mse': abs_mse.item(),
            })
        
    
    avg_loss = total / len(dataloader)
    wandb.log({
        'avg_loss': avg_loss
    })
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    

# Save the trained model
torch.save(model.state_dict(), 'spectral_model10.pth')
print("Training completed and model saved.")