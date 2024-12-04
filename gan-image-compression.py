import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from PIL import Image

# Create output directories
os.makedirs('./compressed_images', exist_ok=True)
os.makedirs('./gan_outputs', exist_ok=True)

# Generator (Decoder) Network
class Generator(nn.Module):
    def __init__(self, latent_dim=256):
        super(Generator, self).__init__()
        
        # Bottleneck representation layers
        self.bottleneck = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU(True)
        )
        
        # Decoder (Upsampling) Network
        self.decoder = nn.Sequential(
            # Initial upsampling layers
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Middle layers
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Final layer
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )
        
    def forward(self, z):
        # Reshape the latent representation
        z = self.bottleneck(z)
        z = z.view(-1, 256, 4, 4)
        
        # Decode the image
        img = self.decoder(z)
        return img

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Input layer
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Hidden layers
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Flatten and classify
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        return self.model(img).view(-1, 1)

# Encoder Network
class Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(Encoder, self).__init__()
        
        # Encoder Network
        self.encoder = nn.Sequential(
            # Initial layers
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            
            # Middle layers
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.Flatten()
        )
        
        # Latent space representation
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encode the input
        features = self.encoder(x)
        
        # Get mean and variance
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        
        # Reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar

# GAN-based Compression Training
def train_gan_compression(epochs=100, batch_size=64, learning_rate=0.0002):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data preparation
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load dataset
    dataset = datasets.ImageFolder(root='./data/train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize networks
    encoder = Encoder().to(device)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Optimizers
    g_optimizer = optim.Adam(list(encoder.parameters()) + list(generator.parameters()), lr=learning_rate, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    # Loss functions
    adversarial_loss = nn.BCELoss()
    reconstruction_loss = nn.MSELoss()
    kl_loss = lambda mu, logvar: torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
    
    # Training loop
    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # Ground truth labels
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)
            
            # Train Encoder and Generator
            g_optimizer.zero_grad()
            
            # Encode real images
            z, mu, logvar = encoder(real_images)
            
            # Reconstruct images
            reconstructed_images = generator(z)
            
            # Adversarial loss
            g_loss_adv = adversarial_loss(discriminator(reconstructed_images), valid)
            
            # Reconstruction loss
            rec_loss = reconstruction_loss(reconstructed_images, real_images)
            
            # KL Divergence loss
            kl_div = kl_loss(mu, logvar)
            
            # Total generator loss
            g_loss = g_loss_adv + rec_loss + 0.001 * kl_div
            g_loss.backward()
            g_optimizer.step()
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            # Real image loss
            real_loss = adversarial_loss(discriminator(real_images), valid)
            
            # Fake image loss
            fake_loss = adversarial_loss(discriminator(reconstructed_images.detach()), fake)
            
            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()
            
            # Print progress
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], '
                      f'G_Loss: {g_loss.item():.4f}, D_Loss: {d_loss.item():.4f}')
        
        # Save sample reconstructions
        with torch.no_grad():
            sample_images = real_images[:8]
            z_sample, _, _ = encoder(sample_images)
            reconstructed_sample = generator(z_sample)
            
            # Denormalize images
            sample_images = sample_images * 0.5 + 0.5
            reconstructed_sample = reconstructed_sample * 0.5 + 0.5
            
            save_image(sample_images, f'./gan_outputs/original_epoch_{epoch+1}.png')
            save_image(reconstructed_sample, f'./gan_outputs/reconstructed_epoch_{epoch+1}.png')
    
    # Save models
    torch.save({
        'encoder': encoder.state_dict(),
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict()
    }, './gan_compression_models.pth')
    
    return encoder, generator, discriminator

# Compression function
def compress_image(image_path, encoder, generator):
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    input_image = transform(image).unsqueeze(0)
    
    # Compress
    with torch.no_grad():
        z, _, _ = encoder(input_image)
        compressed_image = generator(z)
        
        # Denormalize
        compressed_image = compressed_image * 0.5 + 0.5
    
    # Save compressed image
    save_image(compressed_image, './compressed_images/gan_compressed_image.png')
    print("Image compressed successfully!")

# Main execution
if __name__ == "__main__":
    # Train the GAN compression model
    encoder, generator, discriminator = train_gan_compression()
    
    # Example compression (uncomment and provide image path)
    # compress_image('./path/to/your/image.jpg', encoder, generator)
