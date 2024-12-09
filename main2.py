import streamlit as st
import cv2
import numpy as np
from scipy.stats import norm

# Function to calculate noise score from the image
def calculate_noise(image):
    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)  # Adjust the kernel size as needed
    
    # Subtract the blurred image from the original to get the noise
    noise_image = cv2.subtract(image, blurred_image)
    
    # Calculate the noise score (variance of pixel values in the noise image)
    noise_score = np.var(noise_image)
    
    return noise_score

# Function to calculate the probability of the image being real or AI-generated
def calculate_probability(noise_score):
    # Real image parameters (mean and variance)
    real_mean = 63.70
    real_variance = 1822
    
    # AI-generated image parameters (mean and variance)
    ai_mean = 28.22
    ai_variance = 458
    
    # Calculate the PDF for the real and AI-generated images
    real_prob = norm.pdf(noise_score, loc=real_mean, scale=np.sqrt(real_variance))
    ai_prob = norm.pdf(noise_score, loc=ai_mean, scale=np.sqrt(ai_variance))
    
    # Normalize the probabilities to make them sum to 1
    total_prob = real_prob + ai_prob
    real_prob_normalized = real_prob / total_prob
    ai_prob_normalized = ai_prob / total_prob
    
    return real_prob_normalized, ai_prob_normalized

# Streamlit UI components
st.title("Image Noise Analysis: Real vs AI-Generated")

# Image upload
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

# Check if an image has been uploaded
if uploaded_image is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    
    # Display the uploaded image
    st.image(image, channels="GRAY", caption="Uploaded Image", use_container_width=True)
    
    # Calculate the noise score
    noise_score = calculate_noise(image)
    st.write(f"Noise Score: {noise_score}")
    
    # Calculate the probabilities for real and AI-generated image
    real_prob, ai_prob = calculate_probability(noise_score)
    
    # Display the probabilities
    st.write(f"Probability of being a Real Image: {real_prob * 100:.2f}%")
    st.write(f"Probability of being an AI-Generated Image: {ai_prob * 100:.2f}%")
else:
    st.write("Please upload an image to analyze.")
