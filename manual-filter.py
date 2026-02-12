import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from math import log10, sqrt
import os

# -----------------------------
# 1. Image Loading and Helper Functions
# -----------------------------
def load_and_prepare_image(image_path):
    """
    Loads an image from the specified path, converts to grayscale.
    Returns the loaded image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Failed to read image at path: {image_path}")
        
    print(f"Image loaded with dimensions: {img.shape}")
    return img

def show_images(images, titles, cmap='gray'):
    """Helper function to display multiple images in a single plot."""
    n = len(images)
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# -----------------------------
# 2. Manual Convolution
# -----------------------------
def manual_convolve(image, kernel):
    """Performs manual convolution using a given kernel."""
    if image.ndim != 2:
        raise ValueError("manual_convolve expects grayscale image")
    img = image.astype(np.float32)
    k = kernel.astype(np.float32)
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode='reflect')
    out = np.zeros_like(img, dtype=np.float32)
    kflip = np.flip(np.flip(k, 0), 1)
    H, W = img.shape
    for i in range(H):
        for j in range(W):
            region = padded[i:i+kh, j:j+kw]
            out[i, j] = np.sum(region * kflip)
    return np.clip(out, 0, 255).astype(np.uint8)

# -----------------------------
# 3. Kernels
# -----------------------------
def box_kernel(size=3):
    return np.ones((size, size), dtype=np.float32) / (size * size)

def gaussian_kernel(size=5, sigma=1.0):
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return (kernel / np.sum(kernel)).astype(np.float32)

def sharpen_kernel():
    return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

def emboss_kernel():
    return np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)

def sobel_x():
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

def sobel_y():
    return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

# -----------------------------
# 4. Metrics
# -----------------------------
def mae(a, b):
    return np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))

def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32))**2)
    if mse == 0:
        return float('inf')
    return 20 * log10(255.0 / sqrt(mse))

# -----------------------------
# 5. Adaptive Filtering
# -----------------------------
def add_gaussian_noise(image, mean=0, var=0.01):
    """Adds Gaussian noise to an image."""
    sigma = var**0.5
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    noisy_img = image + gaussian_noise
    return np.clip(noisy_img, 0, 1).astype(np.float32)

def wiener_filter(image, local_mean, local_variance, noise_variance):
    """Applies the Wiener filter based on local statistics."""
    # This is a simplified Wiener filter implementation
    local_snr = (local_variance - noise_variance) / local_variance
    local_snr = np.maximum(local_snr, 0)
    filtered_pixel = local_mean + (local_snr * (image - local_mean))
    return np.clip(filtered_pixel, 0, 1)

def adaptive_median_filter(image, max_kernel_size=7):
    """
    Applies an adaptive median filter. The kernel size changes based on
    whether the pixel is a noise outlier.
    """
    img_padded = np.pad(image, max_kernel_size // 2, mode='reflect')
    output = np.zeros_like(image, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            s_max = max_kernel_size
            kernel_size = 3
            while kernel_size <= s_max:
                pad = kernel_size // 2
                window = img_padded[i + max_kernel_size // 2 - pad : i + max_kernel_size // 2 + pad + 1,
                                     j + max_kernel_size // 2 - pad : j + max_kernel_size // 2 + pad + 1]
                
                Z_min = np.min(window)
                Z_max = np.max(window)
                Z_med = np.median(window)
                Z_xy = image[i, j]

                if Z_min < Z_med < Z_max:
                    if Z_min < Z_xy < Z_max:
                        output[i, j] = Z_xy
                    else:
                        output[i, j] = Z_med
                    break
                else:
                    kernel_size += 2
            else:
                output[i, j] = Z_med
    return output

# -----------------------------
# 6. Main Lab Execution
# -----------------------------
def run_all_labs():
    # --- Part A: Load User Image ---
    # REPLACE WITH YOUR IMAGE PATH
    IMAGE_PATH = r'/Users/sharan/Downloads/Lewis Hamilton.jpeg'
    
    try:
        img = load_and_prepare_image(IMAGE_PATH)
    except FileNotFoundError as e:
        print(e)
        print("Using a synthetic image for fallback...")
        def make_test_image(h=360, w=540):
            img = np.zeros((h, w), dtype=np.uint8)
            for i in range(h):
                img[i, :] = np.clip(np.linspace(30, 220, w) + (i / h) * 30, 0, 255)
            cv2.circle(img, (int(w * 0.25), int(h * 0.35)), 60, 20, -1)
            cv2.rectangle(img, (int(w * 0.55), int(h * 0.15)), (int(w * 0.9), int(h * 0.4)), 240, -1)
            cv2.line(img, (0, h - 40), (w, h - 1), 100, 8)
            cv2.putText(img, "Fallback", (int(w * 0.1), int(h * 0.8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, 10, 4, cv2.LINE_AA)
            return img
        img = make_test_image()

    # --- Part B: Manual vs. OpenCV Comparison ---
    print("\n--- Running Manual vs. OpenCV Comparison ---")
    filters = {
        "box3": box_kernel(3),
        "gauss5_s1": gaussian_kernel(5, 1.0),
        "sharpen": sharpen_kernel(),
        "emboss": emboss_kernel(),
        "sobel_x": sobel_x(),
        "sobel_y": sobel_y(),
    }
    
    results = []
    for name, kernel in filters.items():
        # Manual
        t0 = time.perf_counter()
        manual_res = manual_convolve(img, kernel)
        t1 = time.perf_counter()
        manual_time = (t1 - t0) * 1000

        # OpenCV equivalent
        t0 = time.perf_counter()
        if name.startswith("box"):
            cv_res = cv2.blur(img, (kernel.shape[0], kernel.shape[1]))
        elif name.startswith("gauss"):
            cv_res = cv2.GaussianBlur(img, (kernel.shape[0], kernel.shape[1]), kernel.sum())
        elif name == "sharpen" or name == "emboss":
            cv_res = cv2.filter2D(img, -1, kernel)
            if name == "emboss":
                cv_res = np.clip(cv_res + 128, 0, 255).astype(np.uint8)
        elif name == "sobel_x":
            cv_res = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3))
        elif name == "sobel_y":
            cv_res = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3))
        t1 = time.perf_counter()
        cv_time = (t1 - t0) * 1000

        results.append({
            "name": name,
            "manual": manual_res,
            "cv": cv_res,
            "manual_time": manual_time,
            "cv_time": cv_time,
            "mae": mae(manual_res, cv_res),
            "psnr": psnr(manual_res, cv_res)
        })

    print("Comparison summary:")
    print("name, manual_time(ms), cv_time(ms), MAE, PSNR(dB)")
    for r in results:
        print(f"{r['name']}, {r['manual_time']:.2f}, {r['cv_time']:.2f}, {r['mae']:.2f}, {r['psnr']:.2f}")

    for r in results:
        show_images([r["manual"], r["cv"]], [f"{r['name']} - Manual", f"{r['name']} - OpenCV"])
        cv2.imwrite(f"{r['name']}_manual.png", r["manual"])
        cv2.imwrite(f"{r['name']}_cv.png", r["cv"])

    # --- Part C: Advanced Filtering Lab ---
    print("\n--- Running Advanced Filtering Lab ---")
    img_norm = img.astype(np.float32) / 255.0

    # 1. Frequency Domain Filtering
    rows, cols = img_norm.shape
    dft = cv2.dft(np.float32(img_norm), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    D_lpf = 30
    lpf_mask = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(lpf_mask, (cols // 2, rows // 2), D_lpf, (1, 1), -1)
    fshift_lowpass = dft_shift * lpf_mask
    img_lowpass = cv2.magnitude(cv2.idft(np.fft.ifftshift(fshift_lowpass))[:, :, 0], cv2.idft(np.fft.ifftshift(fshift_lowpass))[:, :, 1])

    hpf_mask = 1 - lpf_mask
    fshift_highpass = dft_shift * hpf_mask
    img_highpass = cv2.magnitude(cv2.idft(np.fft.ifftshift(fshift_highpass))[:, :, 0], cv2.idft(np.fft.ifftshift(fshift_highpass))[:, :, 1])

    show_images([img, img_lowpass, img_highpass],
                ["Original", "Low-Pass Filtered", "High-Pass Filtered"])

    # 2. Noise Removal & Edge-Preserving Smoothing
    noisy_img = img.copy()
    noise = np.random.rand(*img.shape)
    noisy_img[noise < 0.02] = 0
    noisy_img[noise > 0.98] = 255
    
    median_filtered = cv2.medianBlur(noisy_img, 5)
    bilateral_filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    show_images([noisy_img, median_filtered, bilateral_filtered],
                ["Noisy Image (Salt & Pepper)", "Median Filtered", "Bilateral Filtered"])

    # --- Part D: Adaptive Filtering ---
    print("\n--- Running Adaptive Filtering ---")
    # Add Gaussian noise for this demonstration
    noisy_gauss_img = add_gaussian_noise(img_norm)
    
    # Simple Adaptive Median Filter
    adaptive_median_res = adaptive_median_filter((noisy_gauss_img * 255).astype(np.uint8))
    
    # Wiener Filter (simulated)
    # Estimate local statistics
    kernel_size = 5
    local_mean = cv2.boxFilter(noisy_gauss_img, -1, (kernel_size, kernel_size))
    local_sq_mean = cv2.boxFilter(noisy_gauss_img**2, -1, (kernel_size, kernel_size))
    local_variance = local_sq_mean - local_mean**2
    
    # Assume global noise variance
    noise_variance = np.var(noisy_gauss_img)
    wiener_res = wiener_filter(noisy_gauss_img, local_mean, local_variance, noise_variance)
    
    show_images([(noisy_gauss_img*255).astype(np.uint8), adaptive_median_res, (wiener_res*255).astype(np.uint8)],
                ["Gaussian Noisy Image", "Adaptive Median Filter", "Wiener Filter"])
    
    # 3. Hybrid Images
    print("\n--- Creating Hybrid Image ---")
    low_freq_path = r'C:\NPCV\gauss5_s1_cv.png'
    high_freq_path = r'C:\NPCV\sobel_y_manual.png'
    
    try:
        img_low, _ = load_and_prepare_image(low_freq_path)
        img_high, _ = load_and_prepare_image(high_freq_path)
        img_low = cv2.resize(img_low, (img_high.shape[1], img_high.shape[0]))
    except FileNotFoundError as e:
        print(f"Error: {e}. Skipping hybrid image task.")
        return

    low_pass_img = cv2.GaussianBlur(img_low, (15, 15), 5)
    high_pass_img = img_high - cv2.GaussianBlur(img_high, (15, 15), 5)

    hybrid_img = low_pass_img + high_pass_img
    hybrid_img = np.clip(hybrid_img, 0, 255).astype(np.uint8)

    show_images([img_low, img_high, hybrid_img],
                ["Low Freq. Component", "High Freq. Component", "Hybrid Image"])

if __name__ == "__main__":
    run_all_labs()