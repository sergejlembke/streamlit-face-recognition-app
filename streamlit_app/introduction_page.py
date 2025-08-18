# --- Third-party imports ---
import streamlit as st

def app() -> None:
    """
    Streamlit introduction page for the Face Recognition App.

    This page provides an overview of the project, its goals, the techniques used (face detection with OpenCV2, face recognition with PCA+SVM and PCA+CS),
    and a navigation guide to the rest of the app. It is intended to orient users and explain what they can learn and try in the app.
    """
    # --- Page Title ---
    st.title('Welcome to the Face Recognition App')
    st.write('')

    # --- Project Overview and Navigation ---
    st.write(
        """
        This project demonstrates a complete pipeline for face detection and face recognition using the Labeled Faces in the Wild (LFW) dataset.
        
        Use the sidebar to navigate between pages and explore the different aspects of face detection and recognition in this project.
        """
    )

    st.write('')    
    col_S1, col_S2, = st.columns([1, 3]) 
    with col_S1:
        st.info('**Navigation Guide**')
    st.write(
        """
        - The **'Eigenimage'** page explains and visualizes eigenimages, showing how PCA captures the main features of faces in the dataset.
        - The **'Webcam'** page lets you test the face detection component live using your own webcam.
        - The **'Model Overview'** page introduces both recognition models and provides a summary of their results and findings. More detailed explanations and results are available on their respective pages (**'Model A'** and **'Model B'**).
        - The **'Application'** page allows you to interactively test the full pipeline: paste a URL of a face image, and see how detection and recognition work together to identify the person.
        """
    )

    st.write('')    
    col_S1, col_S2, = st.columns([1, 3]) 
    with col_S1:
        st.info('**Project Goals**')
    st.write(
        """
        - To explore and compare the effectiveness of a learning-based approach (PCA + SVM) and a non-learning baseline (PCA + CS) for face recognition.
        - To provide an interactive and educational demonstration of face detection and recognition techniques.
        - To visualize and explain the concept of eigenimages (principal components) and their role in face recognition.
        """
    )

    st.write('')    
    col_S1, col_S2, = st.columns([1, 3]) 
    with col_S1:
        st.info('**Techniques Used**')
    st.write(
        """
        - **Face Detection:** Performed using OpenCV2 and the Haar Cascade classifier, which locates faces in images for further processing.
        - **Face Recognition:** Two different models are implemented and compared:
            - **Model A:** Principal Component Analysis (PCA) for dimensionality reduction, followed by a Support Vector Machine (SVM) for supervised classification.
            - **Model B:** Principal Component Analysis (PCA) for dimensionality reduction, followed by Cosine Similarity (CS) as a non-learning baseline for recognition.
        """
    )