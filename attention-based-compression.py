import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import math
import torch.nn.functional as F
from PIL import Image

# Create output directories
os.makedirs('./compressed_images', exist_ok=True)
os.makedirs('./attention_outputs', exist_ok=True)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Multi-Head Self-Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        
        return output
    
    def forward(self, Q, K, V, mask=None):
        batch_size, seq_length, d_model = Q.size()
        
        # Linear projections
        Q = self.W_q(Q).view(batch_size, seq_length, self.num_heads, self.d_k)
        K = self.W_k(K).view(batch_size, seq_length, self.num_heads, self.d_k)
        V = self.W_v(V).view(batch_size, seq_length, self.num_heads, self.d_k)
        
        # Transpose for multi-head attention
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, d_model)
        
        return self.W_o(attn_output)

# Attention-Based Encoder
class AttentionEncoder(nn.Module):
    def __init__(self, image_size=64, channels=3, d_model=256, num_heads=8):
        super(AttentionEncoder, self).__init__()
        
        # Initial convolutional layers
        self.initial_conv = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Flatten and project
        self.flatten = nn.Flatten()
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Multi-Head Attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # Final compression layer
        self.compress = nn.Linear(d_model * (image_size // 8)**2, d_model)
    
    def forward(self, x):
        # Convolutional feature extraction
        features = self.initial_conv(x)
        
        # Flatten and add positional encoding
        features_flat = self.flatten(features)
        features_flat = features_flat.view(features_flat.size(0), -1, features.size(1))
        
        # Add positional encoding
        features_encoded = self.positional_encoding(features_flat)
        
        # Apply attention
        attended_features = self.attention(features_encoded, features_encoded, features_encoded)
        
        # Compress to latent representation
        latent = self.compress(attended_features.view(attended_features.size(0), -1))
        
        return latent

# Attention-Based Decoder
class AttentionDecoder(nn.Module):
    def __init__(self, latent_dim=256, channels=3, image_size=64):
        super(AttentionDecoder, self).__init__()
        
        # Expand latent representation
        self.expand = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, (image_size // 8)**2 * 256)
        )
        
        # Deconvolutional layers for reconstruction
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, latent):
        # Expand latent representation
        expanded = self.expand(latent)
        
        # Reshape for deconvolution
        reshaped = expanded.view(expanded.size(0), 256, 8, 8)
        
        # Reconstruct image
        reconstructed = self.decoder(reshaped)
        
        return reconstructed

# Attention-Based Compression Model
class AttentionCompressionModel(nn.Module):
    def __init__(self, image_size=64, channels=3, latent_dim=256):
        super(AttentionCompressionModel, self).__init__()
        
        self.encoder = AttentionEncoder(image_size, channels, latent_dim)
        self.decoder = AttentionDecoder(latent_dim, channels, image_size)
    
    def forward(self, x):
        # Encode to latent space
        latent = self.encoder(x)
        
        # Decode back to image
        reconstructed = self.decoder(latent)
        
        return reconstructed

# Training Function
def train_attention_compression(epochs=100, batch_size=64, learning_rate=1e-4):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data preparation
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    # Load dataset
    dataset = datasets.ImageFolder(root='./data/train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = AttentionCompressionModel().to(device)
    
    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, _) in enumerate(dataloader):
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
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
        
        # Average loss
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')
        
        # Save sample reconstructions
        with torch.no_grad():
            model.eval()
            sample_images = images[:8]
            reconstructed_sample = model(sample_images)
            
            save_image(sample_images, f'./attention_outputs/original_epoch_{epoch+1}.png')
            save_image(reconstructed_sample, f'./attention_outputs/reconstructed_epoch_{epoch+1}.png')
    
    # Save model
    torch.save(model.state_dict(), './attention_compression_model.pth')
    
    return model

# Image Compression Function
def compress_image(image_path, model=None):
    # Load model if not provided
    if model is None:
        model = AttentionCompressionModel()
        model.load_state_dict(torch.load('./attention_compression_model.pth'))
    
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    input_image = transform(image).unsqueeze(0)
    
    # Compress
    with torch.no_grad():
        model.eval()
        compressed_image = model(input_image)
    
    # Save compressed image
    save_image(compressed_image, './compressed_images/attention_compressed_image.png')
    print("Image compressed successfully!")

# Main execution
if __name__ == "__main__":
    # Train the attention-based compression model
    model = train_attention_compression()
    
    # Example compression (uncomment and provide image path)
    # compress_image('./path/to/the/image.jpg', model)
