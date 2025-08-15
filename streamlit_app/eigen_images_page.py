# --- Third-party imports ---
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.decomposition import PCA

def app() -> None:
    """
    Streamlit page for displaying eigenimages based on PCA.
    """
    st.title('Eigenimage')
    
    # Combined explanation of eigenimages, PCA, and the slider functionality
    st.write(
        """
        This page visualizes the concept of eigenimages using 530 images of George W. Bush from the LFW dataset. By applying Principal Component Analysis (PCA), each face image is represented as a combination of principal components (eigenimages), which capture the most important features across all images and illustrate dimensionality reduction.
        
        Use the slider to select the number of principal components used for reconstruction. When the slider is set to 1, the reconstructed image shows the 'average face' of George W. Bush, as only the first principal component (which captures the most variance) is used. Increasing the number of components adds more detail and individuality to the reconstruction, approaching the original image as more components are included.
        """
    )

    # Load Bush images (uint8) and normalize for PCA
    X_Bush_uint8 = np.load('eigenfaces/Bush.npz')['Bush']
    X_Bush_float64 = X_Bush_uint8.reshape(X_Bush_uint8.shape[0], -1).astype(np.float64) / 255.0

    # Slider to select an image and number of PCA components
    pic = st.slider('Select an image:', 1, 530, 197, key=1) + 1
    n_components = st.slider('Reduce to $n$ components:', 1, 400, 90, key=2)

    # Layout columns for original and eigen images
    col_1, col_2, col_3 = st.columns([1.5, 1.5, 2])

    with col_1:
        # Show the original image
        image_4Darray = X_Bush_uint8[pic]
        fig = plt.figure()
        plt.title('Original Image')
        plt.xlabel(f'$w$')
        plt.ylabel(f'$h$')
        plt.imshow(image_4Darray)
        st.pyplot(fig)

    with col_2:
        # Apply PCA and show the reconstructed eigenimage
        pca_show = PCA(n_components=n_components, whiten=True, random_state=42)
        X_2D_pca = pca_show.fit_transform(X_Bush_float64)
        X_2D_pca_inv = pca_show.inverse_transform(X_2D_pca)
        X_4D_pca_inv = X_2D_pca_inv[pic].reshape(100, 75, 3)
        X_4D_pca_inv_uint8 = (np.clip(X_4D_pca_inv, 0, 1) * 255).astype(np.uint8)
        fig = plt.figure()
        plt.title('Eigenimage')
        plt.xlabel(f'$w$')
        plt.ylabel(f'$h$')
        plt.imshow(X_4D_pca_inv_uint8)
        st.pyplot(fig)