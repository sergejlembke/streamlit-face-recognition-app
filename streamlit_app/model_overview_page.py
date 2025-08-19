# --- Standard library imports ---
import os

# --- Third-party imports ---
import streamlit as st


def app() -> None:
    """
    Streamlit page providing an overview and comparison of the face recognition models in this project.

    The page is intended to help users and readers understand the differences between the models, their performance, and the rationale for
    using both a learning-based and a non-learning baseline in the context of face recognition.
    """
    # --- Page Title ---
    st.title('Face Recognition Model Overview')
        
    col_S1, col_S2, col_S3, col_S4 = st.columns([1.3, 2, 2, 3]) 
    
    with col_S1:
        st.info('Component')
        st.write('Dimensionality Reduction:')
        st.write('Estimator:')
        st.divider() 
        st.info('Training Targets')
        st.write('3 persons:')
        st.write('5 persons:')
        st.write('8 persons:')
        st.write('19 persons:')
    
    with col_S2:
        st.info('Model A: PCA + SVM (Supervised Learning)')
        st.write('Principal Component Analysis (PCA)')
        st.write('Support Vector Machine (SVM)')
        st.divider() 
        st.info('Version: Accuracy')
        st.write('A1: 96.16%')
        st.write('A2: 90.79%')
        st.write('A3: 85.56%')
        st.write('A4: 74.87%')
        
    with col_S3:
        st.info('Model B: PCA + CS (No Learning)')
        st.write('Principal Component Analysis (PCA)')
        st.write('Cosine Similarity (CS)')
        st.divider()
        st.info('Version: Accuracy')
        st.write('B1: 96.15%')
        st.write('B2: 90.79')
        st.write('B3: 87.78%')
        st.write('B4: 77.54%')

    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    
    # Get the absolute path to the file, relative to the current script
    file_path = os.path.join(os.path.dirname(__file__), "plots", "numer_of_pictures.png")
    st.image(image=file_path,
            caption='Distribution of images per person in the LFW dataset',
            width=900)
    
    st.write(
        'Since persons with a large amount of images in the dataset (right edge of the $x$-axis) are the minority, '
        'I decided to use Random Oversampling (ROS) to increase the number of samples for underrepresented classes in the training dataset.'
    )
    
    # --- Summary of Results Section: Model A (PCA + SVM) ---
    st.write('')
    st.write('')
    st.subheader('Summary of Results')
    
    col_S1, col_S2, = st.columns([1, 3]) 
    with col_S1:
        st.info('**Model A (PCA + SVM)**')
    st.write(
        """
        Model A combines Principal Component Analysis (PCA) for dimensionality reduction with a Support Vector Machine (SVM) classifier.

        - The model achieves high classification accuracy, especially when using RGB images, as color information provides valuable features for distinguishing between individuals.
        - Joint hyperparameter tuning of PCA components and SVM parameters (using Gridsearch-CV) leads to optimal performance and robust generalization.
        - The validation accuracy is consistently high, variance across cross-validation folds is generally small, indicating stable and reliable results.
        - The accuracy drops with increasing numbers of classes and more challenging cases.
        - The model is sensitive to variations in lighting, pose, and expression, which can impact performance.
        """
    )
    st.write('')
    st.write('')
    col_S1, col_S2, = st.columns([1, 3]) 
    with col_S1:
        st.info('**Model B (PCA + Cosine Similarity)**')
    st.write(
        """
        The Cosine Similarity (CS) approach provides a simple, non-learning baseline for face recognition by comparing the direction of feature vectors in a high-dimensional space.
        
        - In many cases, the model correctly identifies the person, especially when the facial features are distinctive and well-represented in the training data.
        - A weak point occurs when different people have similar facial features, or when the test image quality or pose differs significantly from the training images.
        - Calculating the CS for the whole dataset can be computationally intensive, as it requires comparing each test image to all training images and then aggregating results per person.
        - The CS method can serve as a fast initial filter or baseline, but may not scale efficiently to very large datasets without further optimization.
        """
    )
    st.write('')
    st.write('')
    col_S1, col_S2, = st.columns([1, 3]) 
    with col_S1:
        st.info('**Comparison**')
    st.write(
        """
        
        
        Overall, Cosine Similarity serves as a useful reference point for evaluating more advanced face recognition models. It highlights the importance of feature representation and the benefits of learning-based approaches for improved accuracy and robustness.

        - Unlike the learning-based Model A, the CS method of Model B does not adapt or optimize for the dataset, so its performance is limited by the raw similarity of the input vectors.
        - As the accuracy of Model B is in the the same range as Model A, it serves as a useful reference point for evaluating more advanced face recognition models.
        - The accuracy of Model A and B is identical for the 3- and 5-person cases.
        - The performance gap widens for the 8- and 19-person cases, highlighting the limitations of both models in complex scenarios. Though Model B outperforms Model A in these cases by a small margin.
        - Computational efficiency is another important consideration. While hyperparameter tuning is needed for Model A to achieve optimal performance, Model B's simpler approach may offer faster inference times in certain situations.
        Though in real-time applications the pre-trained models used in Model A provide a significant advantage in terms of speed and efficiency as can be witnessed in the 'Application' page.
        """
    )