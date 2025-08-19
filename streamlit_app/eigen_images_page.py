# --- Standard library imports ---
import os

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
        To visualize the concept of eigenimages, the 200 images of George W. Bush from the LFW dataset are used.
        By applying Principal Component Analysis (PCA) for dimension reduction, each face image is represented as a combination of principal components (eigenimages), 
        which capture the most important features across all images and illustrate dimensionality reduction.

        Use the slider to select the number of principal components used for reconstruction.
        When the slider is set to 1, the reconstructed image shows the 'average face' of George W. Bush, as only the first principal component (which captures the most variance) is used. Increasing the number of components adds more detail and individuality to the reconstruction, approaching the original image as more components are included.
        """
    )


    # Load the first 100 images of George W. Bush from the LFW dataset
    from lfw_utils import get_lfw_data_cached
    lfw = get_lfw_data_cached(color=True, resize=0.8, funneled=True, download_if_missing=True)
    bush_id = list(lfw.target_names).index('George W Bush')
    bush_mask = lfw.target == bush_id
    X_Bush_uint8 = lfw.images[bush_mask][:100]  # shape: (100, h, w, 3)
    X_Bush_float64 = X_Bush_uint8.reshape(X_Bush_uint8.shape[0], -1).astype(np.float64)

    # Slider to select an image and number of PCA components
    pic = st.slider('Select an image:', 0, X_Bush_uint8.shape[0] - 1, 142, key=1)
    n_components = st.slider('Reduce to $n$ components:', 1, min(100, X_Bush_float64.shape[1]), 100, key=2)

    # Layout columns for original and eigen images
    col_1, col_2, col_3 = st.columns([1.5, 1.5, 2])
    h, w, c = X_Bush_uint8.shape[1:]

    with col_1:
        # Show the original image (use uint8 for display)
        fig = plt.figure()
        plt.title('Original Image')
        plt.xlabel(f'$w$')
        plt.ylabel(f'$h$')
        plt.imshow(X_Bush_uint8[pic])
        st.pyplot(fig)

    with col_2:
        # Apply PCA and show the reconstructed eigenimage
        pca_show = PCA(n_components=n_components, whiten=False, random_state=42)
        X_2D_pca = pca_show.fit_transform(X_Bush_float64)
        X_2D_pca_inv = pca_show.inverse_transform(X_2D_pca)
        X_4D_pca_inv = X_2D_pca_inv[pic].reshape(h, w, c)
        # Plot float image directly for debugging
        fig = plt.figure()
        plt.title('Eigenimage (float)')
        plt.xlabel(f'$w$')
        plt.ylabel(f'$h$')
        plt.imshow(np.clip(X_4D_pca_inv, 0, 1))
        st.pyplot(fig)