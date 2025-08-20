# --- Standard library imports ---
import os

# --- Third-party imports ---
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.decomposition import PCA
from typing import Any

# --- Local imports ---
import face_detection
import face_recognition
from lfw_utils import get_lfw_data_cached


# Fetch the LFW dataset (cached)
lfw = get_lfw_data_cached(color=True, resize=0.8, funneled=True, download_if_missing=True)

def plot_og_marked(og_marked: np.ndarray) -> None:
    """
    Plot the original image with detected face marked.
    """
    fig = plt.figure()
    plt.title('Original image with detected face')
    plt.xlabel(f'$w$')
    plt.ylabel(f'$h$')
    plt.imshow(og_marked)
    st.pyplot(fig)

def plot_cropface(crop: np.ndarray) -> None:
    """
    Plot the cropped face image.
    """
    fig = plt.figure()
    plt.title('Detected face')
    plt.xlabel(f'$w$')
    plt.ylabel(f'$h$')
    plt.imshow(crop)
    st.pyplot(fig)

def plot_eigenface(X_2D: np.ndarray, pca: PCA, sca: Any) -> None:
    """
    Plot the reconstructed eigenface from PCA and StandardScaler.
    X_2D: PCA-transformed face
    pca: fitted PCA object
    sca: fitted StandardScaler object
    """
    # Inverse transform to original feature space
    X_2D_pca_inv = pca.inverse_transform(X_2D)
    X_2D_inv_scaled = sca.inverse_transform(X_2D_pca_inv)
    X_2D_img = (X_2D_inv_scaled * 255).clip(0, 255).astype(np.uint8)
    X_4D_pca_inv = X_2D_img.reshape(100, 75, 3)
    fig = plt.figure()
    plt.title('Eigenimage of detected face')
    plt.xlabel(f'$w$')
    plt.ylabel(f'$h$')
    plt.imshow(X_4D_pca_inv)
    st.pyplot(fig)
    

def app() -> None:
    """
    Main Streamlit app function for face detection and recognition demo.
    """
    # --- Page Title ---
    st.title('Demo of Face Detection & Recognition')
    st.write('')
    st.write('')
    st.write('')
    
    st.write(
    """
    To use this demo:
    - Select a face recognition model (**Model A: PCA + SVM** or **Model B: PCA + CS**), each with different configurations and sets of people trained on
    - See the list of people recognized by each model in the radio button captions
    - Paste the URL of an image containing a person named in the captions (e.g., Colin Powell, George W. Bush, etc.)
    - The app will perform face detection and then apply the selected recognition model to predict the identity in the image
    - The results, including the detected face, the respective eigenimage and predicted name, are displayed interactively
    """
        
    )
    
    
    # --- Model selection and confusion matrix ---
    col_1, col_2 = st.columns([1,1])
    
    with col_1:
        # Radio button for model selection with captions
        selected_model = st.radio(
            "**Select a model for prediction**",
            [
                ':rainbow[A1: PCA + SVM [96.15 % Accuracy]]',
                ':rainbow[B1: PCA + CS [96.15 % Accuracy]]',
                ':rainbow[A2: PCA + SVM [90.79 % Accuracy]]',
                ':rainbow[B2: PCA + CS [90.79 % Accuracy]]',
                ':rainbow[A3: PCA + SVM [85.56 % Accuracy]]',
                ':rainbow[B3: PCA + CS [87.78 % Accuracy]]',
                ':rainbow[A4: PCA + SVM [74.87 % Accuracy]]',
                ':rainbow[B4: PCA + CS [77.54 % Accuracy]]'
            ],
            captions=[
                "",
                "3 people  [Colin Powell, George W Bush, Tony Blair]",
                "",
                "5 people  [Colin Powell, Donald Rumsfeld, George W Bush, Gerhard Schroeder, Tony Blair]",
                "",
                "8 people  [Ariel Sharon, Colin Powell, Donald Rumsfeld, George W Bush, Gerhard Schroeder, Hugo Chavez, Junichiro Koizumi, Tony Blair]",
                "",
                "19 people [Ariel Sharon, Arnold Schwarzenegger, Colin Powell, Donald Rumsfeld, George W Bush, Gerhard Schroeder, Gloria Macapagal Arroyo, Hugo Chavez, Jacques Chirac, Jean Chretien, Jennifer Capriati, John Ashcroft, Junichiro Koizumi, Laura Bush, Lleyton Hewitt, Luiz Inacio Lula da Silva, Serena Williams, Tony Blair, Vladimir Putin]",
            ]
        ) 
    with col_2:
        # Display the corresponding confusion matrix image for the selected model
        if 'A1' in selected_model:
            filename_CM = file_path = os.path.join(os.path.dirname(__file__), "plots", "CM_A_140.png")
        elif 'A2' in selected_model:
            filename_CM = file_path = os.path.join(os.path.dirname(__file__), "plots", "CM_A_100.png")
        elif 'A3' in selected_model:
            filename_CM = file_path = os.path.join(os.path.dirname(__file__), "plots", "CM_A_60.png")
        elif 'A4' in selected_model:
            filename_CM = file_path = os.path.join(os.path.dirname(__file__), "plots", "CM_A_40.png")
        elif 'B1' in selected_model:
            filename_CM = file_path = os.path.join(os.path.dirname(__file__), "plots", "CM_B_140.png")
        elif 'B2' in selected_model:
            filename_CM = file_path = os.path.join(os.path.dirname(__file__), "plots", "CM_B_100.png")
        elif 'B3' in selected_model:
            filename_CM = file_path = os.path.join(os.path.dirname(__file__), "plots", "CM_B_60.png")
        elif 'B4' in selected_model:
            filename_CM = file_path = os.path.join(os.path.dirname(__file__), "plots", "CM_B_40.png")

        st.image(
            image=filename_CM,
            caption=f'Confusion Matrix of the selected model',
            width=600
        )

    # --- Model application ---
    # Input for image URL
    image_url = st.text_input(
        'Insert image URL for detection and recognition',
        'https://www.datocms-assets.com/128928/1742429537-colin-powell-main.jpg?auto=compress%2Cformat&fit=crop&h=640&w=960'
    )
    try: 
        # Face detection on the input image
        og, og_marked, crop = face_detection.get_face(image_url)

        if 'PCA + SVM' in selected_model:
            # Layout for detection and recognition results
            col_1, col_2 = st.columns([1,1])
            col_11, col_12, col_21, col_22 = st.columns([1,1,1,1])

            with col_1:
                st.info('Face Detection')
                with col_11:
                    try:
                        plot_og_marked(og_marked)
                    except:
                        pass
                
                with col_12:
                    try:
                        plot_cropface(crop)
                    except:
                        pass
                
            with col_2:
                st.info('Face Recognition')
                with col_21:
                    with st.spinner('Calculating results...', show_time=True):
                        try:
                            # Apply selected SVM model and plot eigenface
                            pca, sca, svm, X_2D = face_recognition.apply_model_A(selected_model=selected_model, crop=crop)
                            plot_eigenface(X_2D, pca, sca)
                        except:
                            pass
                        
                    with col_22:
                        try:
                            # Predict identity using SVM
                            y_all = lfw.target
                            target_names_all = lfw.target_names
                            y_id = y_all
                            y_names = target_names_all
                            
                            y_pred = svm.predict(X_2D)
                            
                            mask = np.isin(np.unique(y_id), y_pred)
                            
                            st.write(f'Prediction:')       
                            st.write(y_names[mask])
                        except:
                            pass
        
        elif 'PCA + CS' in selected_model:
            # Layout for detection and recognition results (Cosine Similarity)
            col_1, col_2 = st.columns([1,1])
            col_11, col_12, col_21, col_22 = st.columns([1,1,1,1])

            with col_1:
                st.info('Face Detection')
                with col_11:
                    try:
                        plot_og_marked(og_marked)
                    except:
                        pass
                
                with col_12:
                    try:
                        plot_cropface(crop)
                    except:
                        pass
                
            with col_2:
                st.info('Face Recognition')
                with col_21:
                    with st.spinner('Calculating results...', show_time=True):
                        try:
                            # Preprocess and plot eigenface for CS model
                            pca, sca, X_2D, X_train_pca, y_train, y_all, relevant_labels, target_names_all, n_targets = face_recognition.preprocess_cs(selected_model=selected_model, crop=crop)
                            plot_eigenface(X_2D, pca, sca)
                        except:
                            pass
                    
                with col_22:
                    try:
                        # Calculate cosine similarity and display prediction
                        mask = face_recognition.calculate_cs(
                            X_2D=X_2D,
                            X_train_pca=X_train_pca,
                            y_train=y_train,
                            y_all=y_all,
                            relevant_labels=relevant_labels,
                            n_targets=n_targets
                        )

                        st.write(f'Prediction:')       
                        st.write(target_names_all[mask])
                    except:
                        pass
            
        else:
            st.write("No model selected.")
            
    except:
        st.info('No face detected.')