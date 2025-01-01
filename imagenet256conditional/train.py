#there are some hardcoded numbers that I will update, be warned
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
import sys
sys.path.append('COSMOS')

from cosmos_tokenizer.image_lib import ImageTokenizer
wandb.init(project="spectral-model-imagenet", name="training-run")

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
    b,channel, height, width = tensor.shape
    device = tensor.device
            
    indices, flat_indices, _ = create_index_maps_fftshift(height, width)
    indices = indices.to(device)
    flat_indices = flat_indices.to(device)
            
    flattened = torch.zeros(b, channel, height * width, device=device).to(torch.cfloat)
    flattened[:, :, flat_indices] = tensor[ :,:, indices[:, 0], indices[:, 1]]
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

def quantize_complex(target_fft):
    # target_fft shape is [b,c, h*w]
    b=target_fft.shape[0]
    #signed log transform
    real_part = torch.sign(torch.real(target_fft))*torch.log1p(torch.abs(torch.real(target_fft)))  # [b,c, h*w]
    imag_part = torch.imag(target_fft)  # [b,c, h*w]
     
        
        

        
    # Create bins for real and imaginary parts
    # ranges found by max/min of orthonormal fft of dataset
    real_bins = torch.linspace(-7, 7, 1024)
        
    imag_bins = torch.linspace(-7, 7, 1024)
        
    real_bins = real_bins.to(target_fft.device)
    imag_bins = imag_bins.to(target_fft.device)
        
    real_indices = torch.bucketize(real_part, real_bins)
    imag_indices = torch.bucketize(imag_part, imag_bins)
        
    real_indices = torch.clamp(real_indices, 0, 1023)
    imag_indices = torch.clamp(imag_indices, 0, 1023)
        
    real_one_hot = torch.zeros(b,16, 1024, target_fft.shape[2]).to(target_fft.device)
    imag_one_hot = torch.zeros(b,16, 1024, target_fft.shape[2]).to(target_fft.device)
        
    # Fill one-hot encodings
    for c in range(16):
        real_one_hot[:,c, :, :] = F.one_hot(real_indices[:,c], num_classes=1024).float().permute(0,2,1)
        imag_one_hot[:,c, :, :] = F.one_hot(imag_indices[:,c], num_classes=1024).float().permute(0,2,1)

    return real_one_hot, imag_one_hot


model_name = "Cosmos-Tokenizer-CI16x16"
encoder = ImageTokenizer(checkpoint_enc=f'pretrained_ckpts/{model_name}/encoder.jit').to('cuda')

class SpectralImageDataset(torch.utils.data.Dataset):
    def __init__(self, split='train'):
        # Keep existing initialization code
        ds = load_dataset("evanarlian/imagenet_1k_resized_256")
        self.dataset = ds[split]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
        ])
        
        #self.decoder=ImageTokenizer(checkpoint_dec=f'pretrained_ckpts/{model_name}/decoder.jit')

    
        
    def __getitem__(self, index):
        image_data = self.dataset[index]['image']
        condition=self.dataset[index]['label']
        image = self.transform(image_data)
        
        return image,condition

    def __len__(self):
        return len(self.dataset)

# Create dataset and dataloader
dataset = SpectralImageDataset()

dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=128,  
    shuffle=True,
    num_workers=12,
    pin_memory=True,  
    persistent_workers=True  
)




    
#for labels we have to fft them and the goal is to match the magnitude and phase class distributions with cross entropy


lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze',normalize=True).to('cuda')
model_name = "Cosmos-Tokenizer-CI16x16"

decoder=ImageTokenizer(checkpoint_dec=f'pretrained_ckpts/{model_name}/decoder.jit').to('cuda')

def loss_fn(x, labels,image_real):
    x_real = x[0]    # [b,c,1024,x*y]
    x_imag = x[1]    # [b,c,1024,x*y]
    target_real = labels[0]    # [b,c,1024,x*y]
    target_imag = labels[1]    # [b,c,1024,x*y]
    
    b, c, vocab, xy = x_real.shape
    
    # Reshape predictions and targets
    
    x_real = x_real.permute(0, 1, 3, 2).reshape(-1, vocab)      # [b*c*xy, 1024]
    x_imag = x_imag.permute(0, 1, 3, 2).reshape(-1, vocab)      # [b*c*xy, 1024]
    target_real = target_real.permute(0, 1, 3, 2).reshape(-1, vocab)    # [b*c*xy, 1024]
    target_imag = target_imag.permute(0, 1, 3, 2).reshape(-1, vocab)    # [b*c*xy, 1024]

    # Get target indices for cross entropy loss
    target_real_indices = torch.argmax(target_real, dim=-1)    # [b*c*xy]
    target_imag_indices = torch.argmax(target_imag, dim=-1)    # [b*c*xy]
    
    # Classification losses
   
    
    real_loss = F.cross_entropy(x_real, target_real_indices)
    imag_loss = F.cross_entropy(x_imag, target_imag_indices)
    
    # Setup for reconstruction
    real_dist = torch.linspace(-7, 7, 1024).to(x_real.device)
    imag_dist = torch.linspace(-7, 7, 1024).to(x_imag.device)

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
    image_target = torch.sign(target_real_value)*(torch.exp(torch.abs(target_real_value))-1) + 1j * target_imag_value
    
    
    image_target = image_target.reshape(b, c, xy)
    image_target = custom_unflatten(image_target, 16, 9)
    image_target = torch.fft.ifftshift(image_target, dim=2)
    image_target = torch.fft.irfft2(image_target, norm='ortho')
    
    # Reconstruction for prediction
    out_ft = torch.sign(real_values)*(torch.exp(torch.abs(real_values))-1) + 1j * imag_values 
    
    out_ft = out_ft.reshape(b, c, xy)
    x_img = custom_unflatten(out_ft, 16, 9)
    x_img = torch.fft.ifftshift(x_img, dim=2)
    x_img = torch.fft.irfft2(x_img, norm='ortho')

    with torch.no_grad():
        x_img=x_img.to(torch.bfloat16)
        x_img=decoder.decode(x_img)
        x_img=x_img.to(torch.float)
        image_target=image_target.to(torch.bfloat16)
        image_target=decoder.decode(image_target)
        image_target=image_target.to(torch.float)
    # Normalize images
    x_img = (x_img - torch.min(x_img)) / (torch.max(x_img) - torch.min(x_img))
    image_target = (image_target - torch.min(image_target)) / (torch.max(image_target) - torch.min(image_target))
    image_real=(image_real - torch.min(image_real)) / (torch.max(image_real) - torch.min(image_real))
    # LPIPS loss
    lpips_loss = lpips(x_img, image_real)
    
    # MSE losses on continuous values
    real_mse = F.mse_loss(real_values, target_real_value)
    imag_mse = F.mse_loss(imag_values, target_imag_value)
    
    # Total loss
    total_loss = real_loss/math.log(1024) + imag_loss/math.log(1024) #+ lpips_loss
    
    return total_loss, real_loss, imag_loss, real_mse, imag_mse, lpips_loss

num_epochs=1
model = SpectralProcessingModel(16, 256, 12)
#model.load_state_dict(torch.load('spectral_model_latent.pth'))
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
        images,condition = batch
        images = images.to(device)
        with torch.no_grad():
            (encoded,)=encoder.encode(images.to(torch.bfloat16))
            encoded=encoded.to(torch.float)
            
        image_fft = torch.fft.rfft2(encoded, norm='ortho')
        image_fft = torch.fft.fftshift(image_fft, dim=1)
        images=images.to(torch.float)
        # Flatten and quantize
        flattened_fft = custom_flatten(image_fft)
        targets = quantize_complex(flattened_fft) #tuple of tensors of shape b,channel,vocab,seq_len
        
        condition=condition.to(device).to(torch.float)
        optimizer.zero_grad()
        output = model(images,condition)
        total_loss, real_loss, imag_loss, real_mse, imag_mse, lpips_loss = loss_fn(output, targets,images)
        
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

torch.save(model.state_dict(), 'spectral_model_latent.pth')
print("Training completed and model saved.")