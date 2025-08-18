# Uses OpenCV (cv2) and haarcascade_frontalface_default.xml, both licensed under Apache 2.0.

# --- Standard library imports ---
import cv2
import os

# --- Third-party imports ---
import numpy as np
import requests


def get_face(image_url: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Download an image from a URL, detect a face, mark it, and crop it for recognition.
    Returns:
        - original image (RGB)
        - image with detected face marked (RGB)
        - cropped face image (RGB, resized to 75x100)
    """
    # Download the image from the provided URL
    img_data = requests.get(image_url).content
    with open('face_detection/image.jpg', 'wb') as handler:
        handler.write(img_data)
        
    # Load the Haar cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier("face_detection/haarcascade_frontalface_default.xml")

    # Get the directory of this script
    dirname = os.path.dirname(__file__)
    relevant_folder = os.path.join(dirname, "face_detection")

    # Walk through the folder to find images
    for root, dirs, images in os.walk(relevant_folder):
        for image_name in images:
            image_path = os.path.join(root, image_name)
            original_image = cv2.imread(image_path)

            # Process the image if it was loaded successfully
            if original_image is not None:
                # Convert from BGR to RGB and to grayscale
                image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

                # Detect faces in the grayscale image using OpenCV2 and the Haar Cascade classifier.
                # The 'haarcascade_frontalface_default.xml' file contains a pre-trained model for frontal face detection.
                # The parameters scaleFactor=1.3 and minNeighbors=4 were chosen after experimentation, as they provided the best balance
                # between detecting true faces and minimizing false positives for this dataset and application.
                faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.3, minNeighbors=4)

                # Only proceed if exactly one face is detected
                if len(faces) != 1:
                    continue 

                for (x, y, w, h) in faces:
                    # Center of the detected face
                    cx = x + w // 2
                    cy = y + h // 2

                    # Desired aspect ratio: 3:4 (width:height)
                    aspect_w = 3
                    aspect_h = 4

                    # Determine new width and height while keeping the center
                    if w / h > aspect_w / aspect_h:
                        # Too wide, limit by height
                        new_h = h
                        new_w = int(h * aspect_w / aspect_h)
                    else:
                        # Too tall, limit by width
                        new_w = w
                        new_h = int(w * aspect_h / aspect_w)

                    # Calculate new top-left and bottom-right corners
                    x1 = max(cx - new_w // 2, 0)
                    y1 = max(cy - new_h // 2, 0)
                    x2 = min(x1 + new_w, image_rgb.shape[1])
                    y2 = min(y1 + new_h, image_rgb.shape[0])

                    # Adjust if crop goes out of bounds
                    x1 = max(x2 - new_w, 0)
                    y1 = max(y2 - new_h, 0)

                    # Crop and resize to desired output size (e.g., 75x100 for 3:4)
                    roi_color = image_rgb[y1:y2, x1:x2]
                    cropped_face = cv2.resize(roi_color, (75, 100))  # 3:4 output

                    # Mark the detected face on the image
                    marked_face = cv2.rectangle(image_rgb.copy(), (x1, y1), (x2, y2), (255, 0, 150), 4)
                    
                    # Return the results as soon as a face is found
                    return original_image, marked_face, cropped_face

    # If no face is found, return None for all outputs
    return None, None, None
