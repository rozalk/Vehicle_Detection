import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from collections import defaultdict

# Load the VGG16 model (CNN) for feature extraction
model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# Function to check if the file is an image
def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

# Function to extract features from images using VGG16
def extract_features(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))  # Resize image to 224x224
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # Preprocess image for VGG16
        features = model.predict(img_array)
        return features.flatten()  # Flatten the feature array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Function to load dataset with multiple vehicle types
def load_dataset(data_dir):
    features = []
    labels = []
    print("Loading dataset from:", data_dir)

    for label in os.listdir(data_dir):  # Loop through folders (vehicle types)
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue  # Skip if it's not a directory

        for img_file in os.listdir(label_dir):  # Loop through files in each folder
            if not is_image_file(img_file):
                continue  # Skip non-image files

            img_path = os.path.join(label_dir, img_file)
            print(f"Processing image: {img_path}")
            feature = extract_features(img_path)
            if feature is not None:  # Only append valid features
                features.append(feature)
                labels.append(label)  # Use folder name as label (e.g., "car", "truck")

    return np.array(features), np.array(labels)

# Sliding window function to slide across the image
def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

# Non-Maximum Suppression (NMS) with IoU
def non_max_suppression_with_iou(boxes, overlap_threshold=0.4):
    if len(boxes) == 0:
        return []

    picked_boxes = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    sorted_indices = np.argsort(y2)

    while len(sorted_indices) > 0:
        last = len(sorted_indices) - 1
        i = sorted_indices[last]
        picked_boxes.append(i)
        xx1 = np.maximum(x1[i], x1[sorted_indices[:-1]])
        yy1 = np.maximum(y1[i], y1[sorted_indices[:-1]])
        xx2 = np.minimum(x2[i], x2[sorted_indices[:-1]])
        yy2 = np.minimum(y2[i], y2[sorted_indices[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / areas[sorted_indices[:-1]]

        sorted_indices = np.delete(sorted_indices, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

    return picked_boxes

# Vehicle detection with confidence threshold and NMS
def detect_vehicles(image_path, model, svm_classifier, step_size=128, scale_factors=[1.0, 1.5], confidence_threshold=0.6):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None

    original_image = image.copy()
    vehicle_count = defaultdict(int)
    boxes = []
    confidence_scores = []

    for scale in scale_factors:
        resized_image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        for (x, y, window) in sliding_window(resized_image, step_size, (224, 224)):
            if window.shape[0] != 224 or window.shape[1] != 224:
                continue

            img_array = cv2.resize(window, (224, 224))
            img_array = img_to_array(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            features = model.predict(img_array).flatten()

            # Predict the type of vehicle using the SVM classifier
            prediction = svm_classifier.predict([features])[0]
            confidence = svm_classifier.decision_function([features])[0]
            
            # Only accept predictions with a high confidence
            if confidence > confidence_threshold:
                print(f"Predicted label: {prediction} with confidence: {confidence}")
                boxes.append([int(x / scale), int(y / scale), int((x + 224) / scale), int((y + 224) / scale)])
                confidence_scores.append(confidence)

                vehicle_count[prediction] += 1
                # Draw bounding box and label
                cv2.rectangle(original_image, (int(x / scale), int(y / scale)), (int((x + 224) / scale), int((y + 224) / scale)), (0, 255, 0), 2)
                cv2.putText(original_image, prediction, (int(x / scale), int(y / scale) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Convert boxes to NumPy array for NMS
    boxes = np.array(boxes)
    if len(boxes) > 0:
        picked_boxes = non_max_suppression_with_iou(boxes)

        # Draw only the picked boxes after NMS
        for i in picked_boxes:
            box = boxes[i]
            cv2.rectangle(original_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # Save the image with detected vehicles
    output_image_path = "detected_vehicles_output.jpg"
    cv2.imwrite(output_image_path, original_image)

    # Print the count of each vehicle type detected
    for vehicle_type, count in vehicle_count.items():
        print(f"Detected {vehicle_type}: {count}")

    return vehicle_count, output_image_path

if __name__ == "__main__":  # Corrected this line to fix the main execution block
    # Load the dataset
    data_dir = 'd:/project/minor project/project/Vehicle_Detection/dataset'  # Update with your correct dataset path
    X, y = load_dataset(data_dir)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the SVM classifier
    svm_classifier = svm.SVC(kernel='linear', C=1.0, probability=True)  # Enable probability for confidence
    svm_classifier.fit(X_train, y_train)

    # Set the path to the image you want to test
    image_path = 'd:/project/minor project/project/Vehicle_Detection/'  # Example real-time photo path

    # Print the absolute image path to verify it's correct
    print("Absolute image path:", os.path.abspath(image_path))

    # Detect vehicles in a new image with adjusted confidence threshold
    detected_vehicles, output_image = detect_vehicles(image_path, model, svm_classifier, confidence_threshold=0.4)

    # Check if the detection was successful
    if detected_vehicles is not None and output_image is not None:
        print(f"Number of vehicles detected: {detected_vehicles}")
        print(f"Processed image saved to: {output_image}")
    else:
        print("Vehicle detection failed due to image loading error.")
