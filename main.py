import streamlit as st
import numpy as np
from scipy.ndimage import convolve
from PIL import Image
import cv2

# Function to create a Gaussian kernel
def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2)) * 
                      np.exp(-(((x - (size - 1) / 2)**2 + 
                                 (y - (size - 1) / 2)**2) / (2 * sigma**2))),
        (size, size)
    )
    return kernel / np.sum(kernel)

# Function to apply a filter
def apply_filter(image, kernel):
    return convolve(image, kernel, mode='reflect')

# Function to add Gaussian noise
def add_gaussian_noise(image, mean, sigma):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

# Streamlit app
st.title("Gaussian Noise and Edge Detection")
st.write("Explore the effects of Gaussian noise on edge detection.")

# File upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpeg", "jpg", "png"])
if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(uploaded_file)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    
    # Parameters for Gaussian noise
    st.sidebar.header("Gaussian Noise Parameters")
    noise_mean = st.sidebar.slider("Mean", -0.5, 0.5, 0.0, step=0.05)
    noise_sigma = st.sidebar.slider("Variance (Sigma)", 0.0, 0.5, 0.1, step=0.05)

    # Parameters for Gaussian filter
    st.sidebar.header("Gaussian Filter Parameters")
    filter_sigma = st.sidebar.slider("Filter Sigma", 0.1, 5.0, 1.0, step=0.1)
    filter_size = int(2 * np.ceil(2 * filter_sigma) + 1)

    # Create and display the Gaussian kernel
    gaussian_kernel_matrix = gaussian_kernel(filter_size, filter_sigma)
    scaled_kernel = (gaussian_kernel_matrix * 1000).astype(int)

    st.subheader("Gaussian Kernel (Scaled and Integerized)")
    grid_html = "<style> .grid { display: grid; grid-template-columns: repeat(" + \
                f"{filter_size}, 1fr); gap: 5px; text-align: center; font-size: 12px " + \
                ".grid div { padding: 1px; border: 1px solid #ddd; }</style>"
    grid_html += "<div class='grid'>"
    for value in scaled_kernel.flatten():
        grid_html += f"<div>{value}</div>"
    grid_html += "</div>"

    st.markdown(grid_html, unsafe_allow_html=True)
    st.write(f"Total Sum of Kernel Values: {np.sum(scaled_kernel)}")

    # Process the image
    noisy_image = add_gaussian_noise(image, mean=noise_mean, sigma=noise_sigma)
    filtered_image = apply_filter(noisy_image, gaussian_kernel_matrix)

    # Convert images back to 8-bit for OpenCV edge detection
    noisy_image_8bit = (noisy_image * 255).astype(np.uint8)
    filtered_image_8bit = (filtered_image * 255).astype(np.uint8)

    # Apply Canny edge detection
    edges_noisy = cv2.Canny(noisy_image_8bit, 100, 200)
    edges_filtered = cv2.Canny(filtered_image_8bit, 100, 200)

    # Display results
    st.subheader("Gaussian Noise and Edge Detection")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(noisy_image, caption="Noisy Image", use_container_width=True, clamp=True)
        st.image(edges_noisy, caption="Edges (Noisy Image)", use_container_width=True, clamp=True, channels="GRAY")
    with col2:
        st.image(filtered_image, caption="Filtered Image (Gaussian)", use_container_width=True, clamp=True)
        st.image(edges_filtered, caption="Edges (Filtered Image)", use_container_width=True, clamp=True, channels="GRAY")
