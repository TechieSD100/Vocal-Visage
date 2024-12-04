import tensorflow as tf
import cv2
import os
import imageio
from typing import Tuple


# Function to load a video and preprocess its frames
def load_video_for_gif(path: str) -> tf.Tensor:
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        frame = frame[190:236, 80:220]  # Crop the frame
        frames.append(frame)

    cap.release()

    # Stack frames into a single array
    frames = tf.convert_to_tensor(frames, dtype=tf.float32)

    # Normalize frames
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(frames)
    normalized_frames = (frames - mean) / std

    # Convert to uint8 (0–255)
    normalized_frames = tf.clip_by_value(normalized_frames * 255, 0, 255)
    return tf.cast(normalized_frames, tf.uint8).numpy()


# Function to load video and alignment data
def load_gif_data(path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    # Convert Tensor to string
    path = path.numpy().decode("utf-8") if isinstance(path, tf.Tensor) else path

    # Extract file name without extension
    file_name = os.path.splitext(os.path.basename(path))[0]

    # Construct paths for video and alignments
    video_path = os.path.join("..", "data", "s1", f"{file_name}.mpg")

    # Load video frames and alignment tokens
    frames = load_video_for_gif(video_path)
    return frames