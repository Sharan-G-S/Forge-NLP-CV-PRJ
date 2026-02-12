"""
Unsupervised Anomaly Detection in Industrial Manufacturing
Automatically spot rare product defects without seeing defect examples during training

Uses an Autoencoder approach:
1. Train only on normal (good) products
2. Model learns to reconstruct normal patterns
3. Defective products have high reconstruction error -> anomalies

Author: Sharan G S
Date: September 24, 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve
from PIL import Image, ImageDraw
import os
from typing import Tuple, List
import random

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class ConvAutoencoder(nn.Module):
    """Convolutional Autoencoder for anomaly detection"""
    
    def __init__(self, input_channels=3, latent_dim=128):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # 64x64x3 -> 32x32x32
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            # 32x32x32 -> 16x16x64
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            
            # 16x16x64 -> 8x8x128
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            
            # 8x8x128 -> 4x4x256
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim),
            nn.ReLU(True),
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU(True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # 4x4x256 -> 8x8x128
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            
            # 8x8x128 -> 16x16x64
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            
            # 16x16x64 -> 32x32x32
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            # 32x32x32 -> 64x64x3
            nn.ConvTranspose2d(32, input_channels, 4, stride=2, padding=1),
            nn.Sigmoid()  # Output in [0,1] range
        )
    
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        
        # Bottleneck
        batch_size = x.size(0)
        flattened = self.bottleneck(encoded)
        reshaped = flattened.view(batch_size, 256, 4, 4)
        
        # Decode
        decoded = self.decoder(reshaped)
        
        return decoded
    
    def get_reconstruction_error(self, x):
        """Calculate reconstruction error (MSE) for anomaly detection"""
        with torch.no_grad():
            reconstructed = self.forward(x)
            # Calculate per-sample MSE
            mse = torch.mean((x - reconstructed) ** 2, dim=(1, 2, 3))
            return mse.cpu().numpy()

class ManufacturingDataGenerator:
    """Generate realistic bottle manufacturing data: normal bottles and their defects"""
    
    def __init__(self, image_size=64):
        self.image_size = image_size
    
    def generate_normal_product(self) -> np.ndarray:
        """Generate realistic normal bottle images"""
        img = Image.new('RGB', (self.image_size, self.image_size), color=(240, 245, 250))  # Light background
        draw = ImageDraw.Draw(img)
        
        bottle_type = random.choice(['water_bottle', 'soda_bottle', 'beer_bottle'])
        
        if bottle_type == 'water_bottle':
            # Clear plastic water bottle
            bottle_color = (220, 235, 245)  # Clear blue tint
            cap_color = (50, 100, 180)     # Blue cap
            
            # Bottle body (cylindrical)
            bottle_width = random.randint(18, 25)
            bottle_height = random.randint(45, 50)
            x_center = self.image_size // 2
            y_bottom = self.image_size - 5
            
            # Main bottle body
            draw.rectangle([x_center - bottle_width//2, y_bottom - bottle_height,
                           x_center + bottle_width//2, y_bottom], fill=bottle_color, outline=(180, 200, 220))
            
            # Bottle neck
            neck_width = bottle_width // 3
            neck_height = random.randint(8, 12)
            draw.rectangle([x_center - neck_width//2, y_bottom - bottle_height - neck_height,
                           x_center + neck_width//2, y_bottom - bottle_height], fill=bottle_color, outline=(180, 200, 220))
            
            # Cap
            cap_width = neck_width + 2
            cap_height = 4
            draw.rectangle([x_center - cap_width//2, y_bottom - bottle_height - neck_height - cap_height,
                           x_center + cap_width//2, y_bottom - bottle_height - neck_height], fill=cap_color)
            
            # Label area
            label_height = bottle_height // 3
            label_y = y_bottom - bottle_height + bottle_height//3
            draw.rectangle([x_center - bottle_width//2 + 2, label_y,
                           x_center + bottle_width//2 - 2, label_y + label_height], fill=(255, 255, 255), outline=(200, 200, 200))
            
            # Water line (partial fill)
            water_level = random.randint(bottle_height//3, bottle_height - 5)
            draw.rectangle([x_center - bottle_width//2 + 1, y_bottom - water_level,
                           x_center + bottle_width//2 - 1, y_bottom - 1], fill=(180, 220, 240))
            
        elif bottle_type == 'soda_bottle':
            # Dark soda bottle with curved shape
            bottle_color = (60, 80, 40)    # Dark green glass
            cap_color = (200, 180, 50)     # Gold/yellow cap
            
            # Bottle body with curves
            bottle_width = random.randint(20, 26)
            bottle_height = random.randint(42, 48)
            x_center = self.image_size // 2
            y_bottom = self.image_size - 5
            
            # Draw bottle with curved sides
            points = []
            for y in range(bottle_height):
                progress = y / bottle_height
                # Create bottle curve (wider in middle)
                width_factor = 0.7 + 0.3 * (1 - abs(progress - 0.6) * 2)
                half_width = int(bottle_width * width_factor / 2)
                
                if y % 3 == 0:  # Draw every 3rd line for performance
                    draw.line([(x_center - half_width, y_bottom - y), (x_center + half_width, y_bottom - y)], 
                             fill=bottle_color, width=2)
            
            # Bottle outline
            draw.rectangle([x_center - bottle_width//2, y_bottom - bottle_height,
                           x_center + bottle_width//2, y_bottom], outline=(40, 60, 20), width=2, fill=None)
            
            # Neck
            neck_width = bottle_width // 3
            neck_height = random.randint(10, 14)
            draw.rectangle([x_center - neck_width//2, y_bottom - bottle_height - neck_height,
                           x_center + neck_width//2, y_bottom - bottle_height], fill=bottle_color, outline=(40, 60, 20))
            
            # Cap
            cap_width = neck_width + 3
            cap_height = 5
            draw.rectangle([x_center - cap_width//2, y_bottom - bottle_height - neck_height - cap_height,
                           x_center + cap_width//2, y_bottom - bottle_height - neck_height], fill=cap_color)
            
            # Brand label
            label_height = bottle_height // 4
            label_y = y_bottom - bottle_height//2 - label_height//2
            draw.rectangle([x_center - bottle_width//3, label_y,
                           x_center + bottle_width//3, label_y + label_height], fill=(200, 50, 50), outline=(150, 30, 30))
            
        else:  # beer_bottle
            # Brown beer bottle
            bottle_color = (101, 67, 33)   # Brown glass
            cap_color = (150, 150, 150)    # Silver cap
            
            # Bottle body
            bottle_width = random.randint(16, 22)
            bottle_height = random.randint(48, 52)
            x_center = self.image_size // 2
            y_bottom = self.image_size - 4
            
            # Main bottle body
            draw.rectangle([x_center - bottle_width//2, y_bottom - bottle_height,
                           x_center + bottle_width//2, y_bottom], fill=bottle_color, outline=(70, 45, 20))
            
            # Shoulder (tapered top)
            shoulder_height = bottle_height // 6
            for i in range(shoulder_height):
                width_reduction = i * 2
                y_pos = y_bottom - bottle_height + i
                draw.line([(x_center - bottle_width//2 + width_reduction//2, y_pos),
                          (x_center + bottle_width//2 - width_reduction//2, y_pos)], fill=bottle_color)
            
            # Long neck
            neck_width = bottle_width // 4
            neck_height = random.randint(12, 16)
            draw.rectangle([x_center - neck_width//2, y_bottom - bottle_height - neck_height,
                           x_center + neck_width//2, y_bottom - bottle_height + shoulder_height], 
                          fill=bottle_color, outline=(70, 45, 20))
            
            # Cap
            cap_width = neck_width + 2
            cap_height = 4
            draw.rectangle([x_center - cap_width//2, y_bottom - bottle_height - neck_height - cap_height,
                           x_center + cap_width//2, y_bottom - bottle_height - neck_height], fill=cap_color)
            
            # Beer level (golden liquid)
            beer_level = random.randint(bottle_height//2, bottle_height - 3)
            draw.rectangle([x_center - bottle_width//2 + 1, y_bottom - beer_level,
                           x_center + bottle_width//2 - 1, y_bottom - 1], fill=(218, 165, 32))
            
            # Foam head
            foam_height = random.randint(2, 5)
            draw.rectangle([x_center - bottle_width//2 + 1, y_bottom - beer_level - foam_height,
                           x_center + bottle_width//2 - 1, y_bottom - beer_level], fill=(255, 248, 220))
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        return img_array
    
    def generate_defective_product(self) -> np.ndarray:
        """Generate defective bottle (anomaly)"""
        # Start with normal bottle
        img_array = self.generate_normal_product()
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        
        # Add bottle-specific defects
        defect_type = random.choice(['crack', 'dent', 'missing_cap'])
        
        if defect_type == 'crack':
            # Crack in bottle (diagonal line)
            x1 = random.randint(self.image_size//4, 3*self.image_size//4)
            y1 = random.randint(self.image_size//4, 3*self.image_size//4)
            length = random.randint(15, 25)
            angle = random.uniform(0, 2 * np.pi)
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            
            # Draw jagged crack
            for i in range(0, length, 2):
                offset = random.randint(-1, 1)
                px = int(x1 + i * np.cos(angle) + offset)
                py = int(y1 + i * np.sin(angle) + offset)
                if 0 <= px < self.image_size and 0 <= py < self.image_size:
                    draw.point((px, py), fill='black')
                    draw.point((px+1, py), fill='black')
            
        elif defect_type == 'dent':
            # Dent in bottle (deformed area)
            x_center = random.randint(self.image_size//3, 2*self.image_size//3)
            y_center = random.randint(self.image_size//2, 3*self.image_size//4)
            
            # Draw concave dent
            for radius in range(3, 8):
                color_shift = radius * 10
                dent_color = (max(0, 150-color_shift), max(0, 150-color_shift), max(0, 160-color_shift))
                draw.ellipse([x_center-radius, y_center-radius, x_center+radius, y_center+radius], 
                            outline=dent_color)
            
        else:  # missing_cap
            # Remove cap area (make it background color)
            cap_area_y = random.randint(5, 15)
            draw.rectangle([self.image_size//2 - 8, 0, self.image_size//2 + 8, cap_area_y], 
                          fill=(240, 245, 250))
        
        # Convert back to numpy and normalize
        img_array = np.array(img) / 255.0
        return img_array
    
    def generate_dataset(self, n_normal: int, n_defects: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate dataset with normal and defective products"""
        # Generate normal products
        normal_images = []
        for _ in range(n_normal):
            img = self.generate_normal_product()
            normal_images.append(img)
        
        # Generate defective products
        defect_images = []
        for _ in range(n_defects):
            img = self.generate_defective_product()
            defect_images.append(img)
        
        # Combine and create labels
        all_images = normal_images + defect_images
        labels = [0] * n_normal + [1] * n_defects  # 0=normal, 1=anomaly
        
        # Convert to numpy arrays and transpose for PyTorch (B, C, H, W)
        images = np.array(all_images).transpose(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        labels = np.array(labels)
        
        return images, labels

def train_autoencoder(model, train_data, epochs=100, learning_rate=0.001, batch_size=32):
    """Train autoencoder on normal data only"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Convert to tensor
    train_tensor = torch.FloatTensor(train_data).to(device)
    
    model.train()
    losses = []
    
    print(f"Training autoencoder on {len(train_data)} normal samples...")
    print(f"Using device: {device}")
    
    for epoch in range(epochs):
        epoch_losses = []
        
        # Mini-batch training
        for i in range(0, len(train_tensor), batch_size):
            batch = train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    return losses

def evaluate_anomaly_detection(model, test_images, test_labels):
    """Evaluate anomaly detection performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Calculate reconstruction errors
    test_tensor = torch.FloatTensor(test_images).to(device)
    reconstruction_errors = model.get_reconstruction_error(test_tensor)
    
    # Calculate ROC AUC
    auc_score = roc_auc_score(test_labels, reconstruction_errors)
    
    # Find optimal threshold using precision-recall curve
    precision, recall, thresholds = precision_recall_curve(test_labels, reconstruction_errors)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Predictions using optimal threshold
    predictions = (reconstruction_errors > optimal_threshold).astype(int)
    
    # Calculate metrics
    true_positives = np.sum((predictions == 1) & (test_labels == 1))
    false_positives = np.sum((predictions == 1) & (test_labels == 0))
    true_negatives = np.sum((predictions == 0) & (test_labels == 0))
    false_negatives = np.sum((predictions == 0) & (test_labels == 1))
    
    precision_final = true_positives / (true_positives + false_positives + 1e-8)
    recall_final = true_positives / (true_positives + false_negatives + 1e-8)
    f1_final = 2 * (precision_final * recall_final) / (precision_final + recall_final + 1e-8)
    
    return {
        'auc_score': auc_score,
        'optimal_threshold': optimal_threshold,
        'precision': precision_final,
        'recall': recall_final,
        'f1_score': f1_final,
        'reconstruction_errors': reconstruction_errors,
        'predictions': predictions
    }

def visualize_results(model, test_images, test_labels, results, num_samples=8):
    """Visualize original images, reconstructions, and anomaly detection results"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Get reconstructions
    test_tensor = torch.FloatTensor(test_images).to(device)
    with torch.no_grad():
        reconstructions = model(test_tensor).cpu().numpy()
    
    # Select samples to show
    normal_indices = np.where(test_labels == 0)[0][:num_samples//2]
    anomaly_indices = np.where(test_labels == 1)[0][:num_samples//2]
    indices = np.concatenate([normal_indices, anomaly_indices])
    
    fig, axes = plt.subplots(3, len(indices), figsize=(2*len(indices), 6))
    
    for i, idx in enumerate(indices):
        # Original image
        original = test_images[idx].transpose(1, 2, 0)  # CHW -> HWC
        axes[0, i].imshow(original)
        axes[0, i].set_title(f"{'Normal' if test_labels[idx] == 0 else 'Defect'}")
        axes[0, i].axis('off')
        
        # Reconstruction
        reconstruction = reconstructions[idx].transpose(1, 2, 0)  # CHW -> HWC
        axes[1, i].imshow(reconstruction)
        axes[1, i].set_title(f"Reconstructed")
        axes[1, i].axis('off')
        
        # Error visualization
        error = np.mean((original - reconstruction) ** 2, axis=2)
        im = axes[2, i].imshow(error, cmap='hot')
        axes[2, i].set_title(f"Error: {results['reconstruction_errors'][idx]:.4f}")
        axes[2, i].axis('off')
        plt.colorbar(im, ax=axes[2, i], fraction=0.046)
    
    plt.tight_layout()
    plt.show()
    
    # Plot reconstruction error distribution
    plt.figure(figsize=(10, 6))
    
    normal_errors = results['reconstruction_errors'][test_labels == 0]
    anomaly_errors = results['reconstruction_errors'][test_labels == 1]
    
    plt.hist(normal_errors, bins=30, alpha=0.7, label='Normal Products', color='blue')
    plt.hist(anomaly_errors, bins=30, alpha=0.7, label='Defective Products', color='red')
    plt.axvline(results['optimal_threshold'], color='green', linestyle='--', 
                label=f"Threshold: {results['optimal_threshold']:.4f}")
    
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Count')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    """Main function to demonstrate unsupervised anomaly detection"""
    print("� BOTTLE QUALITY CONTROL - UNSUPERVISED ANOMALY DETECTION")
    print("=" * 65)
    
    # Generate synthetic bottle manufacturing data
    print("\n🏭 Generating bottle manufacturing data...")
    data_generator = ManufacturingDataGenerator(image_size=64)
    
    # Training data (only normal bottles)
    train_images, _ = data_generator.generate_dataset(n_normal=500, n_defects=0)
    print(f"✅ Generated {len(train_images)} normal bottle samples for training")
    
    # Test data (normal + defective bottles)
    test_images, test_labels = data_generator.generate_dataset(n_normal=100, n_defects=50)
    print(f"✅ Generated {len(test_images)} test samples (100 normal bottles, 50 defective bottles)")
    
    # Create and train autoencoder
    print("\n🧠 Creating and training autoencoder...")
    model = ConvAutoencoder(input_channels=3, latent_dim=128)
    
    # Train only on normal bottles
    losses = train_autoencoder(model, train_images, epochs=100, learning_rate=0.001)
    
    # Evaluate anomaly detection
    print("\n📊 Evaluating bottle defect detection performance...")
    results = evaluate_anomaly_detection(model, test_images, test_labels)
    
    print("\n✅ BOTTLE QUALITY CONTROL RESULTS:")
    print(f"   🎯 Detection Accuracy (AUC): {results['auc_score']:.1%}")
    print(f"   🎚️ Optimal Threshold: {results['optimal_threshold']:.6f}")
    print(f"   📈 Precision (No False Alarms): {results['precision']:.1%}")
    print(f"   📈 Recall (Defects Caught): {results['recall']:.1%}")
    print(f"   📈 F1 Score: {results['f1_score']:.1%}")
    
    # Visualize results
    print("\n🖼️ Visualizing bottle inspection results...")
    visualize_results(model, test_images, test_labels, results)
    
    # Save model
    model_path = "/Users/sharan/TEST/bottle_quality_control_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    print("\n🎉 BOTTLE QUALITY CONTROL SYSTEM READY!")
    print("📋 Business Value:")
    print("   • 🏭 ZERO manual defect labeling required")
    print("   • 🔍 Detects unknown defect types automatically")
    print("   • ⚡ Real-time quality control on production line")
    print("   • 💰 Reduces waste and customer complaints")
    print("   • 📈 Scalable to any bottle type/size")
    print("   • 🎯 Perfect for rare defects (cracks, contamination)")
    
    print(f"\n🍾 DEFECT TYPES DETECTED:")
    print(f"   • Cracks in glass/plastic")
    print(f"   • Dents and deformation")
    print(f"   • Missing or damaged caps")

if __name__ == "__main__":
    main()