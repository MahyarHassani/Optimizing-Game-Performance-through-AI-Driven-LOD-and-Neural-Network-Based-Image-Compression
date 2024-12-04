import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image

# Create folders for saving outputs
os.makedirs('./compressed_images', exist_ok=True)

# Define the Autoencoder for Image Compression
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.ReLU(True),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.Sigmoid()  # Normalize output between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder():
    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data preparation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images for faster training
        transforms.ToTensor(),
    ])

    # Ensure you have a dataset at this path
    train_dataset = datasets.ImageFolder(root='./data/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, Optimizer
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()  # Measure reconstruction error
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    print("Starting Training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # Forward pass
            reconstructed = model(data)
            loss = criterion(reconstructed, data)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Logging
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # Average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
        # Save reconstructed images for visualization
        model.eval()
        with torch.no_grad():
            if len(train_loader) > 0:
                sample_data = next(iter(train_loader))[0][:8].to(device)
                reconstructed_images = model(sample_data)[:8].detach().cpu()
                
                save_image(reconstructed_images, f'./compressed_images/reconstructed_epoch_{epoch+1}.png')
                save_image(sample_data.cpu(), f'./compressed_images/original_epoch_{epoch+1}.png')

    # Save the trained model
    torch.save(model.state_dict(), './image_compression_autoencoder.pth')
    print("Model saved!")
    
    return model

def test_image_compression(image_path, model=None, output_path='./compressed_images/compressed_image.png'):
    # Load model if not provided
    if model is None:
        model = Autoencoder()
        model.load_state_dict(torch.load('./image_compression_autoencoder.pth'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Prepare image
    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    # Open and transform image
    image = Image.open(image_path).convert('RGB')
    input_image = transform_test(image).unsqueeze(0).to(device)
    
    # Compress and save
    with torch.no_grad():
        compressed_image = model(input_image).detach().cpu()
        save_image(compressed_image, output_path)
    
    print(f"Compressed image saved to {output_path}")
    return compressed_image

# Main execution
if __name__ == "__main__":
    # Train the model
    trained_model = train_autoencoder()
    
    # Example of testing on a single image (uncomment and provide path)
    # test_image_compression('./path/to/your/test/image.jpg')
