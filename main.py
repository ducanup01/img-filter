import streamlit as st
import numpy as np
from scipy.ndimage import convolve
from PIL import Image
import cv2

# Function to create a Gaussian kernel
def gaussian_kernel(size, sigma=1):
    kernel = np.fromfunction(
        lambda x, y: (1/(2 * np.pi * sigma**2)) * 
                      np.exp(-(((x - (size - 1) / 2)**2 + 
                                 (y - (size - 1) / 2)**2) / (2 * sigma**2))),
        (size, size)
    )
    return kernel / np.sum(kernel)

# Function to apply a filter
def apply_filter(image, kernel):
    return convolve(image, kernel, mode='reflect')

# Function to add Gaussian noise
def add_gaussian_noise(image, mean=0, sigma=0.1):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

# Streamlit app
st.title("Image Filtering and Edge Detection")
st.write("Upload an image to apply noise, filters, and edge detection interactively.")

# File upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpeg", "jpg", "png"])
if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = np.array(image) / 255.0  # Normalize to [0, 1]

    # Parameters for noise
    st.sidebar.header("Noise Parameters")
    noise_sigma = st.sidebar.slider("Gaussian Noise Sigma", 0.0, 0.5, 0.1, step=0.05)

    # Parameters for Gaussian filter
    st.sidebar.header("Gaussian Filter Parameters")
    filter_sigma = st.sidebar.slider("Gaussian Filter Sigma", 0.1, 10.0, 5.0, step=0.05)
    filter_size = int(2 * np.ceil(2 * filter_sigma) + 1)

    # Parameters for Mean filter
    st.sidebar.header("Mean Filter Parameters")
    mean_filter_size = st.sidebar.slider("Mean Filter Size", 3, 15, 5, step=1)
                                         

    # Process the image
    noisy_image = add_gaussian_noise(image, sigma=noise_sigma)
    gaussian_kernel_matrix = gaussian_kernel(filter_size, filter_sigma)
    filtered_gaussian = apply_filter(noisy_image, gaussian_kernel_matrix)
    mean_kernel = np.ones((mean_filter_size, mean_filter_size)) / (mean_filter_size ** 2)
    filtered_mean = apply_filter(noisy_image, mean_kernel)

    # Convert images back to 8-bit for OpenCV edge detection
    noisy_image_8bit = (noisy_image * 255).astype(np.uint8)
    filtered_gaussian_8bit = (filtered_gaussian * 255).astype(np.uint8)
    filtered_mean_8bit = (filtered_mean * 255).astype(np.uint8)

    # Apply Canny edge detection
    edges_noisy = cv2.Canny(noisy_image_8bit, 100, 200)
    edges_gaussian = cv2.Canny(filtered_gaussian_8bit, 100, 200)
    edges_mean = cv2.Canny(filtered_mean_8bit, 100, 200)

    # Display results
    st.subheader("Blurred and Edge-Detected Images")
    
    # Layout: Blurred images
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(noisy_image, caption="Noisy Image", use_container_width=True, clamp=True)
    with col2:
        st.image(filtered_gaussian, caption="Gaussian Filtered Image", use_container_width=True, clamp=True)
    with col3:
        st.image(filtered_mean, caption="Mean Filtered Image", use_container_width=True, clamp=True)

    # Layout: Edge-detected images
    col4, col5, col6 = st.columns(3)
    with col4:
        st.image(edges_noisy, caption="Edges (Noisy Image)", use_container_width=True, clamp=True, channels="GRAY")
    with col5:
        st.image(edges_gaussian, caption="Edges (Gaussian Filter)", use_container_width=True, clamp=True, channels="GRAY")
    with col6:
        st.image(edges_mean, caption="Edges (Mean Filter)", use_container_width=True, clamp=True, channels="GRAY")
