import streamlit as st
import numpy as np
from skimage import io, util, img_as_float
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Load the image
def load_image(file):
    img = Image.open(file).convert('L')  # Convert to grayscale
    img_array = img_as_float(np.array(img))  # Convert image to float [0, 1] range
    return img, img_array

# Add Gaussian noise to the image
def add_gaussian_noise(image, mean, var):
    return util.random_noise(image, mode='gaussian', mean=mean, var=var)

def process_image(img, kernel_size, sigma):
    kernel_size = max(3, 2 * kernel_size + 1)  # Ensure odd kernel size
    mean = mean_slider / 100.0  # Mean scaled from the slider
    noisy_img = add_gaussian_noise(img, mean=mean, var=0.01)
    denoised = cv2.GaussianBlur((noisy_img * 255).astype(np.uint8), (kernel_size, kernel_size), sigma)
    edges_noisy = cv2.Canny((noisy_img * 255).astype(np.uint8), 100, 200)
    edges_denoised = cv2.Canny(denoised, 100, 200)
    return noisy_img, denoised, edges_noisy, edges_denoised

# Function to plot pixel intensity insights
def show_pixel_insights(img, kernel_size, sigma, noisy_img, denoised_img):
    kernel_size = max(3, 2 * kernel_size + 1)
    noisy_row = (noisy_img * 255)[noisy_img.shape[0] // 2, :]
    denoised_row = (denoised_img * 255)[noisy_img.shape[0] // 2, :]
    
    x = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian_kernel = np.outer(x, x)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(wspace=0.3)
    
    # Plot pixel intensity profiles
    axes[0].plot(noisy_row, label="Noisy Image", color='red')
    axes[0].plot(denoised_row, label="Denoised Image", color='blue')
    axes[0].set_title("Pixel Intensity Profile (Middle Row)")
    axes[0].legend()
    axes[0].set_xlabel("Pixel Index")
    axes[0].set_ylabel("Intensity")

    # Plot Gaussian kernel
    im = axes[1].imshow(gaussian_kernel, cmap='viridis')
    axes[1].set_title("Gaussian Kernel")
    fig.colorbar(im, ax=axes[1])
    
    canvas = FigureCanvas(fig)
    st.pyplot(fig)

# Streamlit UI
st.title("Gaussian Filter and Edge Detection")

# File upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpeg", "jpg", "png"])
if uploaded_file is not None:
    # Load the uploaded image
    original_img, img = load_image(uploaded_file)
    st.image(original_img, caption='Original Image', use_container_width=True)
    
    # Sliders for parameters
    st.sidebar.header("Gaussian Filter Parameters")
    mean_slider = st.sidebar.slider("Mean (x 0.01)", -50, 50, 0, step=1)
    kernel_slider = st.sidebar.slider("Kernel Size", 1, 10, 3)
    sigma_slider = st.sidebar.slider("Sigma (x 0.1)", 1, 50, 10, step=1)
    
    # Process the image
    noisy_img, denoised_img, edges_noisy, edges_denoised = process_image(img, kernel_slider, sigma_slider / 10.0)

    # Display noisy image and edges
    st.subheader("Noisy Image and Edge Detection")
    col1, col2 = st.columns(2)
    with col1:
        st.image((noisy_img * 255).astype(np.uint8), caption="Noisy Image", use_container_width=True, channels="GRAY")
        st.image(edges_noisy, caption="Edges (Noisy Image)", use_container_width=True, channels="GRAY")
    with col2:
        st.image((denoised_img * 255).astype(np.uint8), caption="Denoised Image", use_container_width=True, channels="GRAY")
        st.image(edges_denoised, caption="Edges (Denoised Image)", use_container_width=True, channels="GRAY")

    # Button to show pixel insights
    if st.button("Show Pixel Insights"):
        show_pixel_insights(img, kernel_slider, sigma_slider / 10.0, noisy_img, denoised_img)

