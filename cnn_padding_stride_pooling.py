"""
Convolutional Neural Network with Padding, Stride, and Pooling
Demonstrates CNN operations with visual outputs and model architecture

Author: Sharan G S
Date: September 24, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
import seaborn as sns

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class CNNModel(nn.Module):
    """
    Comprehensive CNN model demonstrating different padding, stride, and pooling techniques
    """
    
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        
        # First Convolutional Block
        # Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Same padding
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second Convolutional Block with different stride
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Stride=2 reduces size
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third Convolutional Block with no padding
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)  # No padding
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fourth Convolutional Block with different kernel size
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)  # Same padding with 5x5 kernel
        self.bn4 = nn.BatchNorm2d(256)
        
        # Different types of pooling layers
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # Standard max pooling
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)  # Average pooling
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d((4, 4))  # Adaptive pooling to fixed size
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """Forward pass with detailed dimension tracking"""
        print(f"Input shape: {x.shape}")
        
        # First conv block with same padding
        x = F.relu(self.bn1(self.conv1(x)))
        print(f"After conv1 (same padding): {x.shape}")
        
        # Max pooling
        x = self.maxpool(x)
        print(f"After maxpool: {x.shape}")
        
        # Second conv block with stride=2
        x = F.relu(self.bn2(self.conv2(x)))
        print(f"After conv2 (stride=2): {x.shape}")
        
        # Third conv block with no padding
        x = F.relu(self.bn3(self.conv3(x)))
        print(f"After conv3 (no padding): {x.shape}")
        
        # Average pooling
        x = self.avgpool(x)
        print(f"After avgpool: {x.shape}")
        
        # Fourth conv block with 5x5 kernel
        x = F.relu(self.bn4(self.conv4(x)))
        print(f"After conv4 (5x5 kernel): {x.shape}")
        
        # Adaptive average pooling to fixed size
        x = self.adaptive_avgpool(x)
        print(f"After adaptive pooling: {x.shape}")
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        print(f"After flattening: {x.shape}")
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        print(f"Final output shape: {x.shape}")
        return x

def visualize_conv_operations():
    """Demonstrate different convolution operations with visual outputs"""
    
    # Create a sample input
    sample_input = torch.randn(1, 1, 8, 8)  # Batch=1, Channels=1, Height=8, Width=8
    
    print("=== Convolution Operations Demonstration ===")
    print(f"Original input shape: {sample_input.shape}")
    
    # Different convolution configurations
    conv_configs = [
        {"name": "Same Padding", "kernel": 3, "stride": 1, "padding": 1},
        {"name": "Valid Padding", "kernel": 3, "stride": 1, "padding": 0},
        {"name": "Stride 2", "kernel": 3, "stride": 2, "padding": 1},
        {"name": "Large Kernel", "kernel": 5, "stride": 1, "padding": 2},
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, config in enumerate(conv_configs):
        # Create convolution layer
        conv = nn.Conv2d(1, 1, kernel_size=config["kernel"], 
                        stride=config["stride"], padding=config["padding"])
        
        # Apply convolution
        with torch.no_grad():
            output = conv(sample_input)
        
        # Plot input and output
        axes[0, i].imshow(sample_input[0, 0].numpy(), cmap='viridis')
        axes[0, i].set_title(f"Input (8x8)")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(output[0, 0].numpy(), cmap='viridis')
        axes[1, i].set_title(f"{config['name']}\nOutput {output.shape[2]}x{output.shape[3]}")
        axes[1, i].axis('off')
        
        print(f"{config['name']}: {sample_input.shape} -> {output.shape}")
    
    plt.tight_layout()
    plt.savefig('conv_operations.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_pooling_operations():
    """Demonstrate different pooling operations"""
    
    # Create a sample feature map
    sample_input = torch.randn(1, 1, 8, 8)
    
    print("\n=== Pooling Operations Demonstration ===")
    
    # Different pooling operations
    pooling_ops = [
        {"name": "Original", "op": None},
        {"name": "Max Pool (2x2)", "op": nn.MaxPool2d(2, 2)},
        {"name": "Avg Pool (2x2)", "op": nn.AvgPool2d(2, 2)},
        {"name": "Max Pool (3x3)", "op": nn.MaxPool2d(3, 3)},
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for i, pool_config in enumerate(pooling_ops):
        if pool_config["op"] is None:
            output = sample_input
        else:
            with torch.no_grad():
                output = pool_config["op"](sample_input)
        
        axes[i].imshow(output[0, 0].numpy(), cmap='plasma')
        axes[i].set_title(f"{pool_config['name']}\nShape: {output.shape[2]}x{output.shape[3]}")
        axes[i].axis('off')
        
        print(f"{pool_config['name']}: {sample_input.shape} -> {output.shape}")
    
    plt.tight_layout()
    plt.savefig('pooling_operations.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_output_size(input_size, kernel_size, stride, padding):
    """Calculate output size for convolution/pooling operations"""
    return (input_size + 2 * padding - kernel_size) // stride + 1

def demonstrate_size_calculations():
    """Demonstrate how to calculate output sizes"""
    
    print("\n=== Size Calculation Formulas ===")
    print("Output Size = (Input Size + 2*Padding - Kernel Size) / Stride + 1")
    print()
    
    examples = [
        {"input": 32, "kernel": 3, "stride": 1, "padding": 1, "name": "Same Padding"},
        {"input": 32, "kernel": 3, "stride": 1, "padding": 0, "name": "Valid Padding"},
        {"input": 32, "kernel": 3, "stride": 2, "padding": 1, "name": "Stride 2"},
        {"input": 32, "kernel": 5, "stride": 1, "padding": 2, "name": "5x5 Kernel"},
    ]
    
    for example in examples:
        output_size = calculate_output_size(
            example["input"], example["kernel"], 
            example["stride"], example["padding"]
        )
        
        print(f"{example['name']}:")
        print(f"  Input: {example['input']}x{example['input']}")
        print(f"  Kernel: {example['kernel']}, Stride: {example['stride']}, Padding: {example['padding']}")
        print(f"  Output: {output_size}x{output_size}")
        print(f"  Calculation: ({example['input']} + 2*{example['padding']} - {example['kernel']}) / {example['stride']} + 1 = {output_size}")
        print()

def create_and_test_model():
    """Create and test the CNN model"""
    
    print("\n=== Creating and Testing CNN Model ===")
    
    # Create model
    model = CNNModel(num_classes=10).to(device)
    
    # Print model summary
    print("\nModel Architecture:")
    try:
        summary(model, (3, 32, 32))
    except:
        print("torchsummary not available, showing manual summary")
        print(model)
    
    # Create sample input
    sample_input = torch.randn(2, 3, 32, 32).to(device)  # Batch size 2, RGB image 32x32
    
    # Forward pass
    print(f"\n=== Forward Pass with Input Shape {sample_input.shape} ===")
    model.eval()
    with torch.no_grad():
        output = model(sample_input)
    
    return model, output

def visualize_feature_maps(model, sample_input):
    """Visualize feature maps from different layers"""
    
    print("\n=== Visualizing Feature Maps ===")
    
    # Hook to capture intermediate outputs
    feature_maps = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            feature_maps[name] = output.detach()
        return hook
    
    # Register hooks
    model.conv1.register_forward_hook(hook_fn('conv1'))
    model.conv2.register_forward_hook(hook_fn('conv2'))
    model.conv3.register_forward_hook(hook_fn('conv3'))
    model.conv4.register_forward_hook(hook_fn('conv4'))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(sample_input[:1])  # Use only first sample
    
    # Visualize feature maps
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    layers = ['conv1', 'conv2', 'conv3', 'conv4']
    
    for i, layer_name in enumerate(layers):
        feature_map = feature_maps[layer_name][0]  # First sample
        
        for j in range(min(4, feature_map.shape[0])):  # Show first 4 channels
            axes[i, j].imshow(feature_map[j].cpu().numpy(), cmap='viridis')
            axes[i, j].set_title(f'{layer_name} - Channel {j}')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('feature_maps.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_simple_example():
    """Train the model on a simple dataset"""
    
    print("\n=== Training Example on CIFAR-10 ===")
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load a small subset of CIFAR-10 for demonstration
    try:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
        
        # Get a small batch for demonstration
        dataiter = iter(trainloader)
        images, labels = next(dataiter)
        images, labels = images.to(device), labels.to(device)
        
        # Create model
        model = CNNModel(num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Single training step demonstration
        model.train()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f"Training Loss: {loss.item():.4f}")
        print(f"Output shape: {outputs.shape}")
        
        # Show some predictions
        _, predicted = torch.max(outputs.data, 1)
        print(f"Labels: {labels[:10].cpu().numpy()}")
        print(f"Predictions: {predicted[:10].cpu().numpy()}")
        
        return model, trainloader
        
    except Exception as e:
        print(f"Could not load CIFAR-10: {e}")
        print("Creating synthetic data for demonstration...")
        
        # Create synthetic data
        synthetic_data = torch.randn(100, 3, 32, 32)
        synthetic_labels = torch.randint(0, 10, (100,))
        dataset = TensorDataset(synthetic_data, synthetic_labels)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        model = CNNModel(num_classes=10).to(device)
        return model, dataloader

def main():
    """Main function to run all demonstrations"""
    
    print("=" * 60)
    print("CNN WITH PADDING, STRIDE, AND POOLING DEMONSTRATION")
    print("=" * 60)
    
    # 1. Demonstrate convolution operations
    visualize_conv_operations()
    
    # 2. Demonstrate pooling operations
    visualize_pooling_operations()
    
    # 3. Show size calculations
    demonstrate_size_calculations()
    
    # 4. Create and test the model
    model, output = create_and_test_model()
    
    # 5. Visualize feature maps
    sample_input = torch.randn(2, 3, 32, 32).to(device)
    visualize_feature_maps(model, sample_input)
    
    # 6. Training example
    trained_model, dataloader = train_simple_example()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ Convolution operations with different padding and stride demonstrated")
    print("✅ Pooling operations (Max, Average, Adaptive) shown")
    print("✅ Output size calculations explained")
    print("✅ Complete CNN model created and tested")
    print("✅ Feature maps visualized")
    print("✅ Training example provided")
    print("\nGenerated files:")
    print("• conv_operations.png - Convolution operation examples")
    print("• pooling_operations.png - Pooling operation examples") 
    print("• feature_maps.png - Feature map visualizations")

if __name__ == "__main__":
    # Install required packages if needed
    try:
        import torchsummary
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.run(["pip", "install", "torchsummary"], check=True)
    
    main()