# pip install torch torchvision opencv-python facenet-pytorch tensorflow

import torch
import torchvision.models as models
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import os
import json
from torchvision import transforms
import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, LeakyReLU, BatchNormalization, Input

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Global variables for models (lazy loading)
mtcnn = None
meso_model = None
transform = None

class Meso4:
    def __init__(self):
        self.model = self.create_model()

    def create_model(self):
        # Input shape matches the resized images
        inputs = Input(shape=(112, 112, 3))
        
        # First convolutional block
        x = Conv2D(8, (3, 3), padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)  # Output: (56, 56, 8)

        # Second convolutional block
        x = Conv2D(16, (5, 5), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)  # Output: (28, 28, 16)

        # Third convolutional block
        x = Conv2D(32, (5, 5), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)  # Output: (14, 14, 32)

        # Fourth convolutional block
        x = Conv2D(64, (5, 5), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)  # Output: (7, 7, 64)

        # Flatten and Dense layers
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(64)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.5)(x)
        x = Dense(1, activation='sigmoid')(x)

        return Model(inputs=inputs, outputs=x)

    def load_weights(self, path):
        self.model.load_weights(path)
        print(f"Weights loaded from {path}")

def load_models():
    global mtcnn, meso_model, transform
    if mtcnn is None:
        print("Loading MTCNN model...")
        mtcnn = MTCNN(keep_all=True, device=device)
    
    if meso_model is None:
        print("Loading Meso4 model...")
        meso_model = Meso4()
        weights_path = "/Users/mirdabhi/Desktop/Coding/ipd copy/hackanova/Deepfake-detection/models/Meso4_DF.weights.h5"
        if os.path.exists(weights_path):
            meso_model.load_weights(weights_path)
        else:
            print(f"Warning: Weights file not found at {weights_path}")

        # Define preprocessing transformations
        transform = transforms.Compose([
            transforms.Resize((112, 112)),  # Resize to match the input size
            transforms.ToTensor(),
        ])

# Function to detect deepfakes in real-time
def detect_face_distortion(video_path, skip_frames=5):
    load_models()  # Load models if not already loaded
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at path {video_path}")
        return 0, 0

    frame_count = 0
    total_frames = 0
    distorted_faces = 0
    example_abnormal_frame = None  # To store one example of an abnormal frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to reduce processing load
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        # Increment total frame count
        total_frames += 1

        # Convert frame to RGB for MTCNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using MTCNN
        boxes, _ = mtcnn.detect(rgb_frame)
        if boxes is None:
            continue

        # Process each detected face
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = rgb_frame[y1:y2, x1:x2]

            # Preprocess the face for Meso4
            face_resized = cv2.resize(face, (112, 112))
            face_normalized = face_resized / 255.0
            face_expanded = np.expand_dims(face_normalized, axis=0)

            # Perform inference with Meso4
            prediction = meso_model.model.predict(face_expanded)[0][0]

            # If distortion (deepfake) is detected (prediction > 0.5 means fake)
            if prediction > 0.5:
                distorted_faces += 1

                # Save one example of an abnormal frame
                if example_abnormal_frame is None:
                    example_abnormal_frame = frame.copy()

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Distorted Face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cap.release()

    # Display one example of an abnormal frame
    # if example_abnormal_frame is not None:
    #     print("Displaying one example of an abnormal frame:")
    #     cv2.imshow("Example Abnormal Frame", example_abnormal_frame)
    #     cv2.waitKey(0)  # Wait until a key is pressed to close the window
    #     cv2.destroyAllWindows()
    # else:
    #     print("No abnormal frames detected.")
    return total_frames, distorted_faces

