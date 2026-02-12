import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from math import log10, sqrt

# -----------------------------
# 1. Load your own image
# -----------------------------
def load_and_prepare_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    print(f"Image loaded with dimensions: {img.shape}")
    return img

# -----------------------------
# 2. Manual convolution
# -----------------------------
def manual_convolve(image, kernel):
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
# 5. Toolbox Filters
# -----------------------------
filters = {
    "box3": box_kernel(3),
    "box7": box_kernel(7),
    "gauss5_s1": gaussian_kernel(5, 1.0),
    "gauss9_s2": gaussian_kernel(9, 2.0),
    "sharpen": sharpen_kernel(),
    "emboss": emboss_kernel(),
    "sobel_x": sobel_x(),
    "sobel_y": sobel_y(),
}

# -----------------------------
# 6. Run Comparison
# -----------------------------
# Specify the path to your image file here
IMAGE_PATH = r'/Users/sharan/Downloads/Lewis Hamilton.jpeg' 
try:
    img = load_and_prepare_image(IMAGE_PATH)
except FileNotFoundError as e:
    print(e)
    # Fallback to the synthetic image if your image isn't found
    print("Falling back to synthetic image...")
    def make_test_image(h=360, w=540):
        img = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            img[i, :] = np.clip(np.linspace(30, 220, w) + (i / h) * 30, 0, 255)
        cv2.circle(img, (int(w * 0.25), int(h * 0.35)), 60, 20, -1)
        cv2.rectangle(img, (int(w * 0.55), int(h * 0.15)), (int(w * 0.9), int(h * 0.4)), 240, -1)
        cv2.line(img, (0, h - 40), (w, h - 1), 100, 8)
        cv2.putText(img, "QBotix", (int(w * 0.1), int(h * 0.8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, 10, 4, cv2.LINE_AA)
        return img
    img = make_test_image()

results = []

for name, kernel in filters.items():
    # manual
    t0 = time.perf_counter()
    manual_res = manual_convolve(img, kernel)
    t1 = time.perf_counter()
    manual_time = (t1 - t0) * 1000

    # OpenCV equivalent
    t0 = time.perf_counter()
    if name.startswith("box"):
        cv_res = cv2.blur(img, (kernel.shape[0], kernel.shape[1]))
    elif name.startswith("gauss"):
        if name == "gauss5_s1":
            cv_res = cv2.GaussianBlur(img, (5, 5), 1.0)
        elif name == "gauss9_s2":
            cv_res = cv2.GaussianBlur(img, (9, 9), 2.0)
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

# -----------------------------
# 7. Show & Save Results
# -----------------------------
print("Comparison summary:")
print("name, manual_time(ms), cv_time(ms), MAE, PSNR(dB)")
for r in results:
    print(f"{r['name']}, {r['manual_time']:.2f}, {r['cv_time']:.2f}, {r['mae']:.2f}, {r['psnr']:.2f}")

# Plot sample results
for r in results:
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(r["manual"], cmap="gray")
    plt.title(f"{r['name']} - Manual")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(r["cv"], cmap="gray")
    plt.title(f"{r['name']} - OpenCV")
    plt.axis("off")

    plt.show()

# Save images
cv2.imwrite("original_gray.png", img)
for r in results:
    cv2.imwrite(f"{r['name']}_manual.png", r["manual"])
    cv2.imwrite(f"{r['name']}_cv.png", r["cv"])