import cv2
import numpy as np
import tensorflow as tf
import video_utils  # Uses your existing processing logic

# --- Configuration ---
MODEL_PATH = "my_video_model.h5"
CLASSES = ["Real", "Fake"]

def predict_single_video(video_path):
    # 1. Load the trained model
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 2. Process the video using your utils
    print(f"Analyzing video: {video_path}")
    video_tensor = video_utils.load_video(video_path)

    if video_tensor is not None:
        # Add batch dimension: (20, 224, 224, 3) -> (1, 20, 224, 224, 3)
        video_tensor = np.expand_dims(video_tensor, axis=0)
        
        # Normalize (if not done in utils)
        video_tensor = video_tensor / 255.0

        # 3. Make Prediction
        prediction = model.predict(video_tensor)
        
        # Get the class with the highest probability
        class_idx = np.argmax(prediction)
        confidence = prediction[0][class_idx] * 100

        print("-" * 30)
        print(f"RESULT: {CLASSES[class_idx]}")
        print(f"CONFIDENCE: {confidence:.2f}%")
        print("-" * 30)
    else:
        print("Error: Could not process video.")

if __name__ == "__main__":
    # Put the path to a video you want to test here!
    test_video = input("Enter the path to the video file: ")
    predict_single_video(test_video)