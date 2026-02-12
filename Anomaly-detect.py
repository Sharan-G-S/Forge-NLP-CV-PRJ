import os
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# -----------------------------
# Utility Functions
# -----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def create_synthetic_dataset(base_dir="./datasets_synth", n_train=100, n_test=30, img_size=64):
    """
    Create synthetic dataset for bottles:
    - train/normal/ -> normal bottles only
    - test/normal/ and test/defect/
    """
    np.random.seed(42)
    torch.manual_seed(42)

    train_dir = os.path.join(base_dir, "train", "normal")
    test_normal_dir = os.path.join(base_dir, "test", "normal")
    test_defect_dir = os.path.join(base_dir, "test", "defect")

    ensure_dir(train_dir)
    ensure_dir(test_normal_dir)
    ensure_dir(test_defect_dir)

    # Only create if empty
    if len(os.listdir(train_dir)) == 0:
        for i in range(n_train):
            img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            x0, y0 = np.random.randint(10, 20), np.random.randint(10, 20)
            x1, y1 = x0 + np.random.randint(20, 40), y0 + np.random.randint(30, 40)
            draw.rectangle([x0, y0, x1, y1], outline="black", fill="gray")
            img.save(os.path.join(train_dir, f"normal_{i}.png"))

    if len(os.listdir(test_normal_dir)) == 0 and len(os.listdir(test_defect_dir)) == 0:
        for i in range(n_test):
            img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            x0, y0 = np.random.randint(10, 20), np.random.randint(10, 20)
            x1, y1 = x0 + np.random.randint(20, 40), y0 + np.random.randint(30, 40)
            draw.rectangle([x0, y0, x1, y1], outline="black", fill="gray")

            if np.random.rand() > 0.5:
                # Add defect
                for _ in range(np.random.randint(1, 3)):
                    dx0, dy0 = np.random.randint(10, 50), np.random.randint(10, 50)
                    dx1, dy1 = dx0 + np.random.randint(5, 10), dy0 + np.random.randint(2, 5)
                    draw.rectangle([dx0, dy0, dx1, dy1], fill="red")
                img.save(os.path.join(test_defect_dir, f"defect_{i}.png"))
            else:
                img.save(os.path.join(test_normal_dir, f"normal_{i}.png"))

    print(f"Synthetic dataset ensured at {base_dir}")

def save_image_grid(paths, scores, out_folder, topk=9):
    """Save top anomalies as a grid"""
    ensure_dir(out_folder)
    pairs = sorted(zip(paths, scores), key=lambda x: x[1], reverse=True)
    selected = pairs[:topk]
    cols = min(3, len(selected))
    rows = (len(selected) + cols - 1) // cols
    plt.figure(figsize=(4 * cols, 4 * rows))
    for i, (p, s) in enumerate(selected, 1):
        img = Image.open(p).convert("RGB")
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.title(f"{os.path.basename(p)}\nscore={s:.4f}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "top_anomalies.png"))
    plt.close()

def save_anomaly_heatmap(orig_img_path, recon_tensor, out_folder="./outputs"):
    """Save heatmap overlay showing reconstruction error"""
    ensure_dir(out_folder)
    
    transform = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
    orig_img = Image.open(orig_img_path).convert("RGB")
    orig_tensor = transform(orig_img)
    
    # Compute per-pixel MSE
    error_map = ((recon_tensor.squeeze().cpu() - orig_tensor)**2).mean(0)  # shape HxW
    
    # Normalize error map
    error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8)
    
    # Convert to heatmap
    cmap = cm.get_cmap("jet")
    heatmap = cmap(error_map.numpy())  # RGBA
    heatmap = Image.fromarray((heatmap[:, :, :3]*255).astype(np.uint8))
    
    # Overlay heatmap on original
    overlay = Image.blend(orig_img.resize((64,64)), heatmap, alpha=0.5)
    
    # Save
    base_name = os.path.basename(orig_img_path)
    overlay.save(os.path.join(out_folder, f"heatmap_{base_name}"))

# -----------------------------
# Autoencoder Model
# -----------------------------
class ConvAutoencoder(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU()
        )
        self.fc1 = nn.Linear(64 * 8 * 8, z_dim)
        self.fc2 = nn.Linear(z_dim, 64 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.fc1(h)
        h2 = self.fc2(z)
        h2 = h2.view(h2.size(0), 64, 8, 8)
        out = self.decoder(h2)
        return out

# -----------------------------
# Training Autoencoder
# -----------------------------
def train_autoencoder(train_loader, device, epochs=20, z_dim=128, save_dir="./outputs"):
    model = ConvAutoencoder(z_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon = model(imgs)
            loss = criterion(recon, imgs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), os.path.join(save_dir, "autoencoder.pth"))
    return model

# -----------------------------
# Main Function
# -----------------------------
def main():
    base_dir = "./datasets_synth"
    create_synthetic_dataset(base_dir=base_dir)  # Ensure dataset exists
    ensure_dir("./outputs")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Load training dataset (only normal bottles)
    train_data = datasets.ImageFolder(os.path.join(base_dir, "train"), transform=transform)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # Collect test images (normal + defect)
    test_normal = os.path.join(base_dir, "test", "normal")
    test_defect = os.path.join(base_dir, "test", "defect")
    test_paths = []
    if os.path.exists(test_normal):
        test_paths += [os.path.join(test_normal, f) for f in os.listdir(test_normal)]
    if os.path.exists(test_defect):
        test_paths += [os.path.join(test_defect, f) for f in os.listdir(test_defect)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---------------- Autoencoder ----------------
    print("\nTraining Autoencoder...")
    model = train_autoencoder(train_loader, device, epochs=20, save_dir="./outputs")

    model.eval()
    scores, paths = [], []
    with torch.no_grad():
        for path in test_paths:
            img = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            recon = model(tensor)
            loss = torch.mean((recon - tensor) ** 2).item()
            scores.append(loss)
            paths.append(path)

            # Save reconstructions and heatmaps for defects
            if "defect" in path:
                recon_img = transforms.ToPILImage()(recon.squeeze().cpu())
                recon_img.save(os.path.join("./outputs", f"recon_{os.path.basename(path)}"))

                save_anomaly_heatmap(path, recon)

    save_image_grid(paths, scores, "./outputs", topk=9)
    print("Results saved in ./outputs")

if __name__ == "__main__":
    main()
