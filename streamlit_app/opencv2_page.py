# --- Third-party imports ---
import streamlit as st

def app() -> None:
    """
    Streamlit app function for demonstrating OpenCV2 functionalities.
    """
    st.title('Face Detection with OpenCV2')

    # Short introduction for the page
    st.write(
        """
        The face detection is performed with the pre-trained model file `haarcascade_frontalface_default.xml`, which is widely used for detecting frontal faces in images.
    
        **How it works:**
        - OpenCV's `CascadeClassifier` is an object detection algorithm that uses Haar-like features to identify objects (in this case, faces) in images. The classifier is trained on thousands of positive and negative images to learn what a human face looks like.
        - The method `CascadeClassifier.detectMultiScale()` scans the input image at multiple scales and locations, returning rectangles where faces are detected.
        - The function takes several parameters, including:
            - `scaleFactor`: Specifies how much the image size is reduced at each image scale. A value of 1.3 means the image is reduced by 30% at each step, allowing the detector to find faces of different sizes.
            - `minNeighbors`: Specifies how many neighbors each candidate rectangle should have to retain it as a valid face. Higher values help reduce false positives.
        - In this project, `face_cascade.detectMultiScale(image_gray, scaleFactor=1.3, minNeighbors=4)` was chosen after experimentation, as it provided the best balance between detecting true faces and minimizing false positives for this dataset and application.

        For more details and to view the original Haar Cascade XML file, see the [official OpenCV GitHub repository](https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml).
        """
    )
