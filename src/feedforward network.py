import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Create output directories
os.makedirs('./game_image_compression_outputs', exist_ok=True)

class GameImageCompressionDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        """
        Custom dataset for image compression
        
        Args:
            image_paths (list): List of image file paths
            transform (callable, optional): Optional transform to be applied on images
        """
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Read image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        return image

class ImageCompressionNet(nn.Module):
    def __init__(self, compression_ratio=0.1):
        """
        Feedforward Autoencoder for Image Compression
        
        Args:
            compression_ratio (float): Relative size of bottleneck layer
        """
        super(ImageCompressionNet, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: 3 x 256 x 256
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 64 x 128 x 128
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 x 64 x 64
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256 x 32 x 32
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            
            # Bottleneck layer with reduced channels
            nn.Conv2d(256, int(256 * compression_ratio), kernel_size=4, stride=2, padding=1)  # Compressed representation
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Upsampling to restore original dimensions
            nn.ConvTranspose2d(int(256 * compression_ratio), 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            # Final layer to restore original color channels
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Normalize output to [0, 1]
        )
    
    def forward(self, x):
        # Encode
        compressed = self.encoder(x)
        
        # Decode
        reconstructed = self.decoder(compressed)
        
        return reconstructed

def train_image_compression(
    image_paths, 
    epochs=50, 
    batch_size=16, 
    learning_rate=1e-4,
    compression_ratio=0.1
):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Create dataset and dataloader
    dataset = GameImageCompressionDataset(
        image_paths, 
        transform=transform
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    # Initialize model
    model = ImageCompressionNet(compression_ratio=compression_ratio).to(device)
    
    # Loss and Optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for image reconstruction
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=5
    )
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, images in enumerate(dataloader):
            images = images.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed_images = model(images)
            
            # Compute loss
            loss = criterion(reconstructed_images, images)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Logging
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'Step [{batch_idx+1}/{len(dataloader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Average loss and scheduler step
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')
        
        # Save model periodically
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'./game_image_compression_model_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), './final_game_image_compression_model.pth')
    
    return model

def compress_and_visualize_image(image_path, model=None, compression_ratio=0.1):
    """
    Compress an image and visualize results
    
    Args:
        image_path (str): Path to input image
        model (torch.nn.Module, optional): Trained model. If None, will load saved model.
        compression_ratio (float): Compression ratio used during training
    
    Returns:
        torch.Tensor: Compressed image representation
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model if not provided
    if model is None:
        model = ImageCompressionNet(compression_ratio=compression_ratio)
        model.load_state_dict(torch.load('./final_game_image_compression_model.pth'))
    
    model = model.to(device)
    model.eval()
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        reconstructed_tensor = model(input_tensor)
        
        # Convert tensors to numpy for visualization
        original_np = input_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        reconstructed_np = reconstructed_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    
    # Visualize results
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.title('Original Image')
    plt.imshow(original_np)
    plt.axis('off')
    
    plt.subplot(122)
    plt.title('Compressed & Reconstructed')
    plt.imshow(reconstructed_np)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('./game_image_compression_outputs/compression_result.png')
    plt.close()
    
    return reconstructed_tensor

def prepare_game_image_dataset():
    """
    Helper function to prepare dataset paths
    
    Returns:
        list: Paths to game images for compression
    """
    # In a real scenario, you would populate these with actual game texture paths
    image_paths = [
        './Textures/texture1.png',
        './Textures/texture2.png',
        # Add more image paths for training
    ]
    
    return image_paths

# Main Execution
if __name__ == "__main__":
    # Prepare dataset paths
    image_paths = prepare_game_image_dataset()
    
    # Train model
    model = train_image_compression(
        image_paths, 
        epochs=50,  # Adjust as needed
        compression_ratio=0.1  # Adjust compression level
    )
    
    # Example compression (uncomment when you have an image)
    # compressed_image = compress_and_visualize_image('./game_assets/test_texture.png', model)
