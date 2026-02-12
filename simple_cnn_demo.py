"""
Simple CNN with Padding, Stride, and Pooling - Image Example
A straightforward demonstration of CNN operations on a real image

Author: Sharan G S
Date: September 24, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# Make figures crisper by default
plt.rcParams['figure.dpi'] = 150

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Conv layers with different padding and stride
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # Same padding
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # Stride=2 
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)  # No padding
        
        # Pooling layers
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        
        # Final classifier
        self.fc = nn.Linear(64 * 7 * 7, 10)  # Adjust size based on your input
        
    def forward(self, x):
        print(f"Input: {x.shape}")
        
        # Conv1 + ReLU (same padding - size stays same)
        x = F.relu(self.conv1(x))
        print(f"After Conv1 (same padding): {x.shape}")
        
        # MaxPool (reduces size by half)
        x = self.maxpool(x)
        print(f"After MaxPool: {x.shape}")
        
        # Conv2 + ReLU (stride=2 - reduces size)
        x = F.relu(self.conv2(x))
        print(f"After Conv2 (stride=2): {x.shape}")
        
        # Conv3 + ReLU (no padding - reduces size)
        x = F.relu(self.conv3(x))
        print(f"After Conv3 (no padding): {x.shape}")
        
        # AvgPool (reduces size again)
        x = self.avgpool(x)
        print(f"After AvgPool: {x.shape}")
        
        # Flatten for classification
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        print(f"Final output: {x.shape}")
        
        return x

def load_sample_image():
    """Load the Lewis Hamilton image without resizing (keep original clarity)."""
    try:
        img_path = "/Users/sharan/TEST/Lewis Hamilton copy.jpeg"
        img = Image.open(img_path).convert('RGB')
        print("✅ Loaded Lewis Hamilton image")
        return img
    except Exception as e:
        print(f"❌ Error loading Lewis Hamilton image: {e}")
        # Create a synthetic colorful image as fallback
        print("📱 Creating synthetic sample image as fallback")
        img_array = np.random.rand(256, 256, 3) * 255
        
        # Add some patterns to make it more interesting
        for i in range(256):
            for j in range(256):
                img_array[i, j, 0] = 255 * (i / 256)  # Red gradient
                img_array[i, j, 1] = 255 * (j / 256)  # Green gradient  
                img_array[i, j, 2] = 128 + 127 * np.sin(i * j / 100)  # Blue pattern
        
        img = Image.fromarray(img_array.astype('uint8'))
        return img

def image_to_tensor(img):
    """Convert PIL image to tensor for the CNN.
    Note: The CNN expects 64x64 input because of the final linear layer dimensions.
    We'll resize a COPY here while keeping the original image intact for clear display.
    """
    # Resize a copy for the model, keep original untouched for display
    img_resized = img.resize((64, 64), resample=Image.BICUBIC)

    # Convert to tensor and normalize
    img_array = np.array(img_resized) / 255.0  # Normalize to 0-1
    img_tensor = torch.tensor(img_array, dtype=torch.float32)

    # Rearrange dimensions: (H, W, C) -> (C, H, W)
    img_tensor = img_tensor.permute(2, 0, 1)

    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor

def visualize_individual_filters(img, model):
    """Show the output of each individual filter one by one"""
    
    # Convert image to tensor
    img_tensor = image_to_tensor(img)
    
    print("\n=== Individual Filter Outputs ===")
    print("📸 Applying each filter one by one...")
    
    model.eval()
    with torch.no_grad():
        current_input = img_tensor
        
        # Step 1: Show all Conv1 filters individually
        print("\n🔍 Step 1: Conv1 Filters (16 filters)")
        conv1_out = F.relu(model.conv1(current_input))
        
        # Show first 8 filters from Conv1
        fig1, axes1 = plt.subplots(2, 4, figsize=(16, 8))
        fig1.suptitle('Conv1 Individual Filters (Same Padding) - First 8 of 16 filters', fontsize=16)
        
        for i in range(8):
            ax = axes1[i//4, i%4]
            feature_map = conv1_out[0, i].numpy()
            im = ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f'Filter {i+1}\n{feature_map.shape}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
        input("Press Enter to continue to MaxPool effect...")  # Pause for user interaction
        
        # Step 2: MaxPool effect
        print("\n🔽 Step 2: MaxPool Effect")
        pool1_out = model.maxpool(conv1_out)
        
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
        fig2.suptitle('MaxPool Effect on Feature Maps', fontsize=16)
        
        # Before pooling
        axes2[0].imshow(conv1_out[0, 0].numpy(), cmap='viridis')
        axes2[0].set_title(f'Before MaxPool\n{conv1_out.shape[2]}x{conv1_out.shape[3]}')
        axes2[0].axis('off')
        
        # After pooling
        axes2[1].imshow(pool1_out[0, 0].numpy(), cmap='viridis')
        axes2[1].set_title(f'After MaxPool\n{pool1_out.shape[2]}x{pool1_out.shape[3]}')
        axes2[1].axis('off')
        
        # Difference visualization
        # Resize for comparison
        before_resized = F.interpolate(conv1_out[0:1, 0:1], size=(pool1_out.shape[2], pool1_out.shape[3]), mode='bilinear')
        diff = torch.abs(before_resized[0, 0] - pool1_out[0, 0]).numpy()
        axes2[2].imshow(diff, cmap='hot')
        axes2[2].set_title('Information Lost\n(Absolute Difference)')
        axes2[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        input("Press Enter to continue to Conv2 filters...")  # Pause for user interaction
        
        # Step 3: Conv2 filters (stride=2 effect)
        print("\n🔍 Step 3: Conv2 Filters (Stride=2) - 32 filters")
        conv2_out = F.relu(model.conv2(pool1_out))
        
        fig3, axes3 = plt.subplots(2, 4, figsize=(16, 8))
        fig3.suptitle('Conv2 Individual Filters (Stride=2) - First 8 of 32 filters', fontsize=16)
        
        for i in range(8):
            ax = axes3[i//4, i%4]
            feature_map = conv2_out[0, i].numpy()
            im = ax.imshow(feature_map, cmap='plasma')
            ax.set_title(f'Filter {i+1}\n{feature_map.shape}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
        input("Press Enter to continue to Conv3 filters...")  # Pause for user interaction
        
        # Step 4: Conv3 filters (no padding effect)
        print("\n🔍 Step 4: Conv3 Filters (No Padding) - 64 filters")
        conv3_out = F.relu(model.conv3(conv2_out))
        
        fig4, axes4 = plt.subplots(2, 4, figsize=(16, 8))
        fig4.suptitle('Conv3 Individual Filters (No Padding) - First 8 of 64 filters', fontsize=16)
        
        for i in range(8):
            ax = axes4[i//4, i%4]
            feature_map = conv3_out[0, i].numpy()
            im = ax.imshow(feature_map, cmap='inferno')
            ax.set_title(f'Filter {i+1}\n{feature_map.shape}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
        input("Press Enter to continue to final feature maps...")  # Pause for user interaction
        
        # Step 5: Final AvgPool effect
        print("\n🔽 Step 5: AvgPool Effect")
        pool2_out = model.avgpool(conv3_out)
        
        fig5, axes5 = plt.subplots(2, 4, figsize=(16, 8))
        fig5.suptitle('Final Feature Maps after AvgPool - First 8 of 64 channels', fontsize=16)
        
        for i in range(8):
            ax = axes5[i//4, i%4]
            feature_map = pool2_out[0, i].numpy()
            im = ax.imshow(feature_map, cmap='magma')
            ax.set_title(f'Channel {i+1}\n{feature_map.shape}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
        print("\n🎉 All individual filter outputs displayed!")

def visualize_operations(img, model):
    """Show the progressive transformation through the network"""
    
    # Convert image to tensor
    img_tensor = image_to_tensor(img)
    
    print("\n=== CNN Operations - Progressive Transformation ===")
    print("📸 Processing image through network...")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
    
    # Show progressive transformation
    plt.figure(figsize=(20, 12))
    
    # Original image
    plt.subplot(2, 6, 1)
    plt.imshow(img)
    plt.title('Original Image\n64x64x3')
    plt.axis('off')
    
    # Manual operations for visualization
    with torch.no_grad():
        # Conv1 (same padding)
        conv1_out = F.relu(model.conv1(img_tensor))
        
        plt.subplot(2, 6, 2)
        # Show RGB-like visualization of first 3 channels
        rgb_vis = torch.stack([conv1_out[0, 0], conv1_out[0, 1], conv1_out[0, 2]], dim=2)
        rgb_vis = (rgb_vis - rgb_vis.min()) / (rgb_vis.max() - rgb_vis.min())  # Normalize
        plt.imshow(rgb_vis.numpy())
        plt.title(f'Conv1 (Same Padding)\n{conv1_out.shape[2]}x{conv1_out.shape[3]}x{conv1_out.shape[1]}')
        plt.axis('off')
        
        # MaxPool
        pool1_out = model.maxpool(conv1_out)
        
        plt.subplot(2, 6, 3)
        feature_map2 = pool1_out[0, 0].numpy()
        plt.imshow(feature_map2, cmap='viridis')
        plt.title(f'After MaxPool\n{pool1_out.shape[2]}x{pool1_out.shape[3]}x{pool1_out.shape[1]}')
        plt.axis('off')
        
        # Conv2 (stride=2)
        conv2_out = F.relu(model.conv2(pool1_out))
        
        plt.subplot(2, 6, 4)
        feature_map3 = conv2_out[0, 0].numpy()
        plt.imshow(feature_map3, cmap='plasma')
        plt.title(f'Conv2 (Stride=2)\n{conv2_out.shape[2]}x{conv2_out.shape[3]}x{conv2_out.shape[1]}')
        plt.axis('off')
        
        # Conv3 (no padding)
        conv3_out = F.relu(model.conv3(conv2_out))
        
        plt.subplot(2, 6, 5)
        feature_map4 = conv3_out[0, 0].numpy()
        plt.imshow(feature_map4, cmap='inferno')
        plt.title(f'Conv3 (No Padding)\n{conv3_out.shape[2]}x{conv3_out.shape[3]}x{conv3_out.shape[1]}')
        plt.axis('off')
        
        # AvgPool
        pool2_out = model.avgpool(conv3_out)
        
        plt.subplot(2, 6, 6)
        feature_map5 = pool2_out[0, 0].numpy()
        plt.imshow(feature_map5, cmap='magma')
        plt.title(f'After AvgPool\n{pool2_out.shape[2]}x{pool2_out.shape[3]}x{pool2_out.shape[1]}')
        plt.axis('off')
        
        # Second row - Show averaged channels and statistics
        plt.subplot(2, 6, 7)
        # Show channel diversity in Conv1
        channel_std = torch.std(conv1_out[0], dim=[1,2]).numpy()
        plt.bar(range(min(16, len(channel_std))), channel_std[:16])
        plt.title('Conv1 Channel\nActivation Variance')
        plt.xlabel('Channel')
        plt.ylabel('Std Dev')
        
        plt.subplot(2, 6, 8)
        # Show channel diversity in Conv2
        channel_std2 = torch.std(conv2_out[0], dim=[1,2]).numpy()
        plt.bar(range(min(16, len(channel_std2))), channel_std2[:16])
        plt.title('Conv2 Channel\nActivation Variance')
        plt.xlabel('Channel')
        plt.ylabel('Std Dev')
        
        plt.subplot(2, 6, 9)
        # Show channel diversity in Conv3
        channel_std3 = torch.std(conv3_out[0], dim=[1,2]).numpy()
        plt.bar(range(min(16, len(channel_std3))), channel_std3[:16])
        plt.title('Conv3 Channel\nActivation Variance')
        plt.xlabel('Channel')
        plt.ylabel('Std Dev')
        
        plt.subplot(2, 6, 10)
        # Average across all channels for final visualization
        avg_features = torch.mean(pool2_out[0], dim=0).numpy()
        plt.imshow(avg_features, cmap='hot')
        plt.title('Final Features\n(All Channels Avg)')
        plt.axis('off')
        
        plt.subplot(2, 6, 11)
        # Final prediction
        predictions = F.softmax(output, dim=1)[0].numpy()
        plt.bar(range(len(predictions)), predictions)
        plt.title('Final Predictions\n(10 classes)')
        plt.xlabel('Class')
        plt.ylabel('Probability')
        
        plt.subplot(2, 6, 12)
        # Show size reduction progression
        sizes = [64*64*3, conv1_out.numel()//conv1_out.shape[0], 
                pool1_out.numel()//pool1_out.shape[0], conv2_out.numel()//conv2_out.shape[0],
                conv3_out.numel()//conv3_out.shape[0], pool2_out.numel()//pool2_out.shape[0]]
        labels = ['Input', 'Conv1', 'Pool1', 'Conv2', 'Conv3', 'Pool2']
        plt.semilogy(range(len(sizes)), sizes, 'o-', linewidth=2, markersize=8)
        plt.title('Feature Map Size\nReduction')
        plt.xlabel('Layer')
        plt.ylabel('Total Elements (log)')
        plt.xticks(range(len(labels)), labels, rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    input("Press Enter to see individual filter outputs one by one...")
    
    # Now show individual filters
    visualize_individual_filters(img, model)

def visualize_single_filter_steps(img, model):
    """Display one figure at a time showing:
    - Original image
    - One feature map after Conv1 (padding effect)
    - One feature map after MaxPool (pooling effect)
    - One feature map after Conv2 with stride=2 (stride effect)
    """
    img_tensor = image_to_tensor(img)
    model.eval()

    with torch.no_grad():
        # 1. Original image (natural resolution)
        plt.figure(figsize=(7, 7))
        plt.imshow(img, interpolation='lanczos')
        plt.title(f'Original Image ({img.size[0]}x{img.size[1]})')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # 2. Padding (Conv1 with same padding)
        conv1_out = F.relu(model.conv1(img_tensor))
        fm_pad = conv1_out[0, 0].cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.imshow(fm_pad, cmap='viridis', interpolation='nearest')
        plt.title(f'Padding: Conv1 (same)\nFeature map 0 — {fm_pad.shape[0]}x{fm_pad.shape[1]}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # 3. Pooling (MaxPool)
        pool1_out = model.maxpool(conv1_out)
        fm_pool = pool1_out[0, 0].cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.imshow(fm_pool, cmap='magma', interpolation='nearest')
        plt.title(f'Pooling: MaxPool(2x2)\nFeature map 0 — {fm_pool.shape[0]}x{fm_pool.shape[1]}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # 4. Stride (Conv2 with stride=2)
        conv2_out = F.relu(model.conv2(pool1_out))
        fm_stride = conv2_out[0, 0].cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.imshow(fm_stride, cmap='plasma', interpolation='nearest')
        plt.title(f'Stride: Conv2 (stride=2)\nFeature map 0 — {fm_stride.shape[0]}x{fm_stride.shape[1]}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def demonstrate_padding_stride_pooling():
    """Simple demonstration of each operation"""
    
    print("\n=== Understanding Each Operation ===")
    
    # Create a simple 6x6 input
    sample_input = torch.randn(1, 1, 6, 6)
    
    print(f"Original input shape: {sample_input.shape}")
    
    # 1. Convolution with different padding
    print("\n1️⃣ PADDING EFFECTS:")
    
    # Same padding
    conv_same = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    out_same = conv_same(sample_input)
    print(f"   Same padding (padding=1): {sample_input.shape} → {out_same.shape}")
    
    # No padding  
    conv_valid = nn.Conv2d(1, 1, kernel_size=3, padding=0)
    out_valid = conv_valid(sample_input)
    print(f"   No padding (padding=0): {sample_input.shape} → {out_valid.shape}")
    
    # 2. Stride effects
    print("\n2️⃣ STRIDE EFFECTS:")
    
    # Stride 1
    conv_s1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
    out_s1 = conv_s1(sample_input)
    print(f"   Stride=1: {sample_input.shape} → {out_s1.shape}")
    
    # Stride 2
    conv_s2 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
    out_s2 = conv_s2(sample_input)
    print(f"   Stride=2: {sample_input.shape} → {out_s2.shape}")
    
    # 3. Pooling effects
    print("\n3️⃣ POOLING EFFECTS:")
    
    # MaxPool
    maxpool = nn.MaxPool2d(2, 2)
    out_max = maxpool(out_same)
    print(f"   MaxPool(2x2): {out_same.shape} → {out_max.shape}")
    
    # AvgPool
    avgpool = nn.AvgPool2d(2, 2)
    out_avg = avgpool(out_same)
    print(f"   AvgPool(2x2): {out_same.shape} → {out_avg.shape}")

def main():
    """Main function"""
    
    print("🧠 SIMPLE CNN WITH PADDING, STRIDE, AND POOLING")
    print("=" * 50)
    
    # Load sample image
    img = load_sample_image()
    print(f"📸 Loaded image size: {img.size}")
    
    # Create model
    model = SimpleCNN()
    
    # Only show one example each for padding, pooling, and stride (no saving)
    print(f"\n🔄 Showing single-filter outputs (padding, pooling, stride)...")
    visualize_single_filter_steps(img, model)
    
    # Summary
    print("\n✅ OPERATIONS DEMONSTRATED:")
    print("   🔲 Padding (Conv1, same) – one feature map")
    print("   🔻 Pooling (MaxPool 2x2) – one feature map")
    print("   ⚡ Stride (Conv2, stride=2) – one feature map")
    print("\n🖼️ Displayed interactively using: /Users/sharan/TEST/Lewis Hamilton copy.jpeg")

if __name__ == "__main__":
    main()