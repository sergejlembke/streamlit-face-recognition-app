# Uses OpenCV (cv2) and haarcascade_frontalface_default.xml, both licensed under Apache 2.0.

# --- Third-party imports ---
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from typing import Optional, Any

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier("face_detection/haarcascade_frontalface_default.xml")

# Initialize session state for storing the captured image
if "captured_image" not in st.session_state:
    st.session_state.captured_image = None

class VideoProcessor(VideoTransformerBase):
    """
    Video processor class for real-time face detection in webcam stream.
    """
    def __init__(self) -> None:
        self.frame: Optional[np.ndarray] = None

    def transform(self, frame: Any) -> np.ndarray:
        """
        Detect faces in the video frame and draw rectangles around them.
        """
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        self.frame = img
        return img

def app() -> None:
    """
    Streamlit app function for live webcam face detection and cropping.
    """
    st.markdown(
        """
        <div class="camera-container">
            <h2>📷 Live Camera with Face Detection</h2>
            <p>Faces are automatically detected and highlighted.</p>
        </div>
        """, unsafe_allow_html=True
    )

    # Start the webcam stream with real-time face detection
    ctx = webrtc_streamer(
        key="camera",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # When the user clicks the capture button
    if ctx.video_processor:
        if st.button("📸 Capture Image"):
            image = ctx.video_processor.frame
            if image is not None:
                # Save the captured image
                st.session_state.captured_image = image
                cv2.imwrite("webcam_snapshot.jpg", image)
                st.success("Image captured and saved as webcam_snapshot.jpg.")

                # Convert to grayscale and detect faces
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)
                if len(faces) > 0:
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
                        x2 = min(x1 + new_w, image.shape[1])
                        y2 = min(y1 + new_h, image.shape[0])

                        # Adjust if crop goes out of bounds
                        x1 = max(x2 - new_w, 0)
                        y1 = max(y2 - new_h, 0)

                        # Crop and resize to 3:4 aspect ratio (e.g., 75x100)
                        face_img = image[y1:y2, x1:x2]
                        resized_face = cv2.resize(face_img, (75, 100))  # 3:4 output
                        cv2.imwrite("face.jpg", resized_face)
                        st.image(cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB), caption="Face", width=200)
                else:
                    st.warning("No face detected.")
            else:
                st.warning("No image captured. Please try again.")