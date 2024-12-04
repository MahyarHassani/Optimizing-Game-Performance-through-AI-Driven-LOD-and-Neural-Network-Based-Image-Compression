import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image

# Create output directories
os.makedirs('./texture_segmentation_outputs', exist_ok=True)

# Custom Dataset for Texture Segmentation
class TextureSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        """
        Custom dataset for texture segmentation
        
        Args:
            image_paths (list): List of image file paths
            mask_paths (list): List of corresponding mask file paths
            transform (callable, optional): Optional transform to be applied on images
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Read image and mask
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms if specified
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Convert to tensors
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(mask).long()
        
        return image, mask

# Texture Segmentation Network with Attention
class TextureSegmentationNet(nn.Module):
    def __init__(self, num_classes=5):
        super(TextureSegmentationNet, self).__init__()
        
        # Encoder (Downsampling)
        self.encoder = nn.Sequential(
            # Initial conv block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Attention Module
        self.attention = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Decoder (Upsampling)
        self.decoder = nn.Sequential(
            # Upsampling block
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Second upsampling block
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Final classification layer
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        
        # Attention mechanism
        attention_map = self.attention(features)
        
        # Apply attention
        features_attended = features * attention_map
        
        # Decoder
        segmentation_map = self.decoder(features_attended)
        
        return segmentation_map

# Training Function
def train_texture_segmentation(
    image_paths, 
    mask_paths, 
    num_classes=5, 
    epochs=50, 
    batch_size=16, 
    learning_rate=1e-4
):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data transforms (using albumentations for more advanced augmentations)
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Create dataset and dataloader
    dataset = TextureSegmentationDataset(
        image_paths, 
        mask_paths, 
        transform=transform
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    # Initialize model
    model = TextureSegmentationNet(num_classes=num_classes).to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
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
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, masks)
            
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
            torch.save(model.state_dict(), f'./texture_segmentation_model_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), './final_texture_segmentation_model.pth')
    
    return model

# Texture Segmentation Inference
def segment_texture(image_path, model, num_classes=5):
    # Prepare model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model if not provided
    if isinstance(model, str):
        model = TextureSegmentationNet(num_classes=num_classes)
        model.load_state_dict(torch.load(model))
    
    model = model.to(device)
    model.eval()
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        
        # Get segmentation map
        _, predicted = torch.max(output, 1)
        segmentation_map = predicted.squeeze().cpu().numpy()
    
    # Visualize segmentation
    plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.title('Original Image')
    plt.imshow(image)
    plt.subplot(132)
    plt.title('Segmentation Map')
    plt.imshow(segmentation_map, cmap='viridis')
    plt.subplot(133)
    plt.title('Overlay')
    plt.imshow(image)
    plt.imshow(segmentation_map, alpha=0.5, cmap='viridis')
    plt.savefig('./texture_segmentation_outputs/segmentation_result.png')
    plt.close()
    
    return segmentation_map

# Example Usage Function
def prepare_texture_segmentation_dataset():
    """
    Helper function to prepare dataset paths
    
    Returns:
        tuple: Lists of image and mask paths
    """
    # In a real scenario, you would populate these lists with actual paths
    image_paths = [
        './data/textures/image1.jpg',
        './data/textures/image2.jpg',
        # Add more image paths
    ]
    
    mask_paths = [
        './data/textures/mask1.png',
        './data/textures/mask2.png',
        # Add more mask paths corresponding to images
    ]
    
    return image_paths, mask_paths

# Main Execution
if __name__ == "__main__":
    # Prepare dataset paths
    image_paths, mask_paths = prepare_texture_segmentation_dataset()
    
    # Train model
    model = train_texture_segmentation(
        image_paths, 
        mask_paths, 
        num_classes=5,  # Adjust based on your texture classes
        epochs=50
    )
    
    # Example segmentation (uncomment when you have an image)
    # segment_texture('./path/to/test/image.jpg', model)
