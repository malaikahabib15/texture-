import cv2
import numpy as np
import random
from skimage.feature import hog

# Function to calculate texture using HOG (Histogram of Oriented Gradients)
def calculate_hog_texture(image):
    if len(image.shape) == 3:  # Convert to grayscale if it's a color image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute HOG features and visualization without the multichannel argument
    hog_features, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), visualize=True)
    
    # Normalize HOG image for display
    hog_image = cv2.normalize(hog_image, None, 0, 255, cv2.NORM_MINMAX)
    hog_image = np.uint8(hog_image)  # Convert to 8-bit
    
    return hog_features, hog_image

# Function to compute Euclidean distance between two feature vectors
def calculate_feature_distance(features1, features2):
    return np.linalg.norm(features1 - features2)

# Function to resize images to a common size
def resize_image(image, size=(512, 512)):
    return cv2.resize(image, size)

# Define a class to handle images and texture comparison for each category
class ImageClass:
    def __init__(self, class_name, image_paths):
        self.class_name = class_name
        self.images = []
        for img_path in image_paths:
            image = cv2.imread(img_path)
            if image is not None:
                self.images.append(image)
            else:
                print(f"Failed to load image: {img_path}")
    
    def get_random_image(self):
        return random.choice(self.images)

# Initialize three classes with example images
class1 = ImageClass("Class 1", ['cloud 1.jpg', 'cloud 2.jpg', 'cloud 3.jpg'])
class2 = ImageClass("Class 2", ['grapes 1.jpg', 'grapes 2.jpg', 'grapes 3.jpg'])
class3 = ImageClass("Class 3", ['parrot 1.jpeg', 'parrot 1.jpg', 'parrot 2.jpg'])

# Step 1: Load the reference image and resize it
reference_image_path = 'cloud 1.jpg'  # Reference image
reference_image = cv2.imread(reference_image_path)
reference_image_resized = resize_image(reference_image)
reference_hog_features, reference_hog_texture = calculate_hog_texture(reference_image_resized)

# Step 2: Randomly select a class
random_class = random.choice([class1, class2, class3])
print(f"Randomly selected class: {random_class.class_name}")

# Step 3: Randomly select an image from the chosen class
random_image = random_class.get_random_image()
random_image_resized = resize_image(random_image)
random_hog_features, random_hog_texture = calculate_hog_texture(random_image_resized)

# Calculate distance between the reference HOG features and the random image HOG features
distance = calculate_feature_distance(reference_hog_features, random_hog_features)
print(f"Distance between textures (HOG): {distance}")

# Define a threshold for similarity
threshold = 0.4  # Adjust based on your needs

# Step 4: Display the results
if distance < threshold:
    print(f"The reference image belongs to {random_class.class_name} category (similar to selected image).")
else:
    print(f"The reference image does not belong to the random image.")

# Optional: Display the reference image and randomly selected image along with their textures
cv2.imshow("Reference Image", reference_image_resized)
cv2.imshow("Random Image from " + random_class.class_name, random_image_resized)
cv2.imshow("Reference HOG Texture", reference_hog_texture)
cv2.imshow("Random Image HOG Texture", random_hog_texture)

# Wait for user interaction
cv2.waitKey(0)
cv2.destroyAllWindows()
