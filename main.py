import os
import numpy as np
import cv2  # Import OpenCV
from sklearn import svm
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image

# Define the path to the dataset
data_dir = '../Vehicle_Detection/dataset'

# Load the VGG16 model (CNN) for feature extraction
model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# Function to check if the file is an image
def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

# Function to extract features from images
def extract_features(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        return features.flatten()
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Function to load dataset
def load_dataset(data_dir):
    features = []
    labels = []
    print("Loading dataset from:", data_dir)
    
    if not os.path.exists(data_dir):
        print(f"Dataset directory does not exist: {data_dir}")
        return np.array(features), np.array(labels)

    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue

        print(f"Processing directory: {label_dir}")
        
        for img_file in os.listdir(label_dir):
            if not is_image_file(img_file):
                continue

            img_path = os.path.join(label_dir, img_file)
            try:
                # Attempt to open the image to check if it's valid
                with Image.open(img_path) as img:
                    img.verify()  # Verify the image file
            except (IOError, SyntaxError) as e:
                print(f"Skipping invalid image file {img_path}: {e}")
                continue

            print(f"Processing image: {img_path}")
            feature = extract_features(img_path)
            if feature is not None:
                features.append(feature)
                labels.append(label)

    return np.array(features), np.array(labels)

# Load the dataset
X, y = load_dataset(data_dir)

if len(X) == 0 or len(y) == 0:
    print("Error: No data loaded. Check your dataset path and contents.")
    exit(1)

# Convert labels to binary (0 for two_wheelers, 1 for four_wheelers)
label_mapping = {'two_wheelers': 0, 'four_wheelers': 1}
y = np.array([label_mapping.get(label, -1) for label in y])

# Check if dataset contains multiple classes
unique_labels = np.unique(y)
if len(unique_labels) <= 1:
    raise ValueError("Dataset does not contain multiple classes. Please ensure your dataset is properly labeled and contains more than one class.")
else:
    print("Dataset contains multiple classes:", unique_labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if training data contains multiple classes
if len(np.unique(y_train)) <= 1:
    raise ValueError("Training data does not contain multiple classes. Please check your data split.")

# Train the SVM classifier
svm_classifier = svm.SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train, y_train)

# Function to perform vehicle detection on a single image
def sliding_window(image, step_size, window_size):
    # Implement the sliding window function
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def detect_vehicles(image_path, model, svm_classifier, window_size=(224, 224), step_size=32):
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist.")
        return {'Two-Wheeler': 0, 'Four-Wheeler': 0}

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return {'Two-Wheeler': 0, 'Four-Wheeler': 0}

    vehicle_count = {'Two-Wheeler': 0, 'Four-Wheeler': 0}

    # Iterate over the sliding windows
    for (x, y, window) in sliding_window(image, step_size, window_size):
        if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
            print(f"Skipping window at ({x}, {y}) due to shape mismatch: {window.shape}")
            continue

        img_array = cv2.resize(window, window_size)
        img_array = img_to_array(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Extract features using VGG16
        features = model.predict(img_array).flatten()
        print(f"Extracted features shape: {features.shape}")

        # Predict using SVM
        prediction = svm_classifier.predict([features])[0]
        print(f"Prediction: {prediction}")

        # Label based on prediction
        label = 'Four-Wheeler' if prediction == 1 else 'Two-Wheeler'
        vehicle_count[label] += 1

        # Draw bounding box on detected vehicles
        color = (0, 255, 0) if prediction == 1 else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x + window_size[0], y + window_size[1]), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the detection output
    output_path = 'output_detection.jpg'
    cv2.imwrite(output_path, image)
    print(f"Total vehicles detected: Two-Wheelers: {vehicle_count['Two-Wheeler']}, Four-Wheelers: {vehicle_count['Four-Wheeler']}")
    return vehicle_count

# Test the model on the single image
image_path = '../Vehicle_Detection/dataset/two_wheelers/1030.jpg'
vehicle_count = detect_vehicles(image_path, model, svm_classifier)
print(f"Results for {image_path}:")
print(f"Two-Wheelers detected: {vehicle_count['Two-Wheeler']}")
print(f"Four-Wheelers detected: {vehicle_count['Four-Wheeler']}")
