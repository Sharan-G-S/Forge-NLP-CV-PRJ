import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

# -----------------------------
# Helper Functions
# -----------------------------
def show_images(images, titles):
    """Helper function to display multiple images in a single plot."""
    n = len(images)
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def load_image(image_path):
    """
    Loads an image from the specified path, converts it to grayscale,
    and normalizes pixel values to the range [0, 1].
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Failed to read image at path: {image_path}")
        
    # Normalize pixel values to [0, 1] for floating point operations
    img_norm = img.astype(np.float32) / 255.0
    return img_norm, img

# -----------------------------
# Main Lab Code
# -----------------------------
def run_image_lab():
    # --- Part 1: Frequency Domain Filtering ---
    print("--- Running Frequency Domain Filtering ---")
    
    # REPLACE WITH YOUR IMAGE PATH
    image_path =r'/Users/sharan/Downloads/Lewis Hamilton.jpeg'    
    try:
        img_norm, img = load_image(image_path)
    except (FileNotFoundError, IOError) as e:
        print(f"Error: {e}. Please check the image path.")
        return

    rows, cols = img_norm.shape
    dft = cv2.dft(np.float32(img_norm), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Low-Pass Filter (LPF)
    D_lpf = 30
    lpf_mask = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(lpf_mask, (cols // 2, rows // 2), D_lpf, (1, 1), -1)
    fshift_lowpass = dft_shift * lpf_mask
    img_lowpass = cv2.magnitude(cv2.idft(np.fft.ifftshift(fshift_lowpass))[:, :, 0], cv2.idft(np.fft.ifftshift(fshift_lowpass))[:, :, 1])

    # High-Pass Filter (HPF)
    hpf_mask = 1 - lpf_mask
    fshift_highpass = dft_shift * hpf_mask
    img_highpass = cv2.magnitude(cv2.idft(np.fft.ifftshift(fshift_highpass))[:, :, 0], cv2.idft(np.fft.ifftshift(fshift_highpass))[:, :, 1])

    show_images([img, img_lowpass, img_highpass],
                ["Original", "Low-Pass Filtered", "High-Pass Filtered"])

    # --- Part 2: Noise Removal & Edge-Preserving Smoothing ---
    print("\n--- Running Denoising and Smoothing ---")
    
    # Add salt & pepper noise to the image
    noisy_img = img.copy()
    noise = np.random.rand(*img.shape)
    noisy_img[noise < 0.02] = 0
    noisy_img[noise > 0.98] = 255
    
    # Denoise with Median Filter
    median_filtered = cv2.medianBlur(noisy_img, 5)

    # Denoise with Bilateral Filter (on original image)
    bilateral_filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    show_images([noisy_img, median_filtered, bilateral_filtered],
                ["Noisy Image (Salt & Pepper)", "Median Filtered", "Bilateral Filtered"])

    # --- Part 3: Hybrid Images ---
    print("\n--- Creating Hybrid Image ---")
    
    # REPLACE THESE WITH YOUR IMAGE PATHS
    low_freq_image_path = r'C:\NPCV\gauss5_s1_cv.png'
    high_freq_image_path = r'C:\NPCV\sobel_y_manual.png'
    
    try:
        img1_norm, img1 = load_image(low_freq_image_path)
        img2_norm, img2 = load_image(high_freq_image_path)
        img2_norm = cv2.resize(img2_norm, (img1_norm.shape[1], img1_norm.shape[0]))
    except (FileNotFoundError, IOError) as e:
        print(f"Error: {e}. Skipping hybrid image task.")
        return

    # Low-pass version of img1
    low_pass_img1 = cv2.GaussianBlur(img1_norm, (15, 15), 5)

    # High-pass version of img2
    high_pass_img2 = img2_norm - cv2.GaussianBlur(img2_norm, (15, 15), 5)

    # Create the hybrid image
    hybrid_img = low_pass_img1 + high_pass_img2
    hybrid_img = np.clip(hybrid_img, 0, 1)

    show_images([img1, img2, hybrid_img],
                ["Low Freq. Component", "High Freq. Component", "Hybrid Image"])

if __name__ == "__main__":
    run_image_lab()