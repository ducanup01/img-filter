import os
import cv2  # OpenCV for image processing
import numpy as np

# Path to the folder containing the images
folder_path = "/home/ducanup01/projects/gaussian/ai_image"  # Replace with your folder path

# Initialize a list to store noise scores
noise_scores = []

# Loop through all images
for i in range(1, 13):  # Assuming images are named ai1.jpg, ai2.jpg, ..., ai30.jpg
    image_path = os.path.join(folder_path, f"ai{i}.png")  # Replace .jpg with the correct extension if needed
    # Read the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Image ai{i} could not be loaded. Skipping...")
        continue
    
    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)  # You can adjust the kernel size (5,5) as needed
    
    # Subtract the blurred image from the original to get the noise
    noise_image = cv2.subtract(image, blurred_image)
    
    # Calculate the noise score (variance of pixel values in the noise image)
    noise_score = np.var(noise_image)
    noise_scores.append(noise_score)

    print(f"Image ai{i}: Noise Score = {noise_score}")

# Calculate mean and variance of the noise scores
if noise_scores:
    mean_noise = np.mean(noise_scores)
    variance_noise = np.var(noise_scores)
    print("\nOverall Results:")
    print(f"Mean Noise Score: {mean_noise}")
    print(f"Variance of Noise Scores: {variance_noise}")
else:
    print("No images were processed successfully.")
