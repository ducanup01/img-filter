import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from PIL import Image

def gaussian_kernel(size, sigma=1):
    kernel = np.fromfunction(
        lambda x, y: (1/(2 * np.pi * sigma**2)) * 
                      np.exp(-(((x - (size - 1) / 2)**2 + 
                                 (y - (size - 1) / 2)**2) / (2 * sigma**2))),
        (size, size)
    )
    return kernel / np.sum(kernel)

def apply_filter(image, kernel):
    return convolve(image, kernel, mode='reflect')

def add_gaussian_noise(image, mean=0, sigma=0.1):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

# Load an image using PIL
image = Image.open('image/meme.jpeg').convert('L')  # Replace with your image path
image = np.array(image) / 255.0  # Normalize to [0, 1]

# Add Gaussian noise
noisy_image = add_gaussian_noise(image, sigma=0.1)

# Apply Gaussian filter
sigma = 5
kernel = gaussian_kernel(int(2 * np.ceil(2 * sigma) + 1), sigma)
filtered_gaussian = apply_filter(noisy_image, kernel)

# Apply Mean filter for comparison
mean_kernel = np.ones((5, 5)) / 25
filtered_mean = apply_filter(noisy_image, mean_kernel)

# Plot the results
plt.figure(figsize=(15, 10))
plt.subplot(1, 3, 1)
plt.title("Noisy Image")
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Gaussian Filtered Image")
plt.imshow(filtered_gaussian, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Mean Filtered Image")
plt.imshow(filtered_mean, cmap='gray')
plt.axis('off')

plt.show()
