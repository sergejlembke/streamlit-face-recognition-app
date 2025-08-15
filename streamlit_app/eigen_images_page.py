# --- Third-party imports ---
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.decomposition import PCA

def app():
    """Main Streamlit app function for displaying eigen images based on PCA."""
    st.title('Eigen Images')
    
    # Description of PCA and eigenimages
    st.write(
        'Reduction of images to $n$ components. The inverse of the PCA can be displayed as an eigenimage (representation of the eigenvectors) to illustrate dimensionality reduction:'
    )
    st.write(
        "The eigen-images shown here are based on a PCA applied to the 530 images of George W. Bush from the LFW dataset. "
        "If the value of the slider for the $n$ dimensions is set to 1, a face of George W. Bush appears, which can be seen as the 'average face' based on the 530 face images of George W. Bush."
    )

    # Load Bush images (uint8) and normalize for PCA
    X_Bush_uint8 = np.load('eigenfaces/Bush.npz')['Bush']
    X_Bush_float64 = X_Bush_uint8.reshape(X_Bush_uint8.shape[0], -1).astype(np.float64) / 255.0

    # Slider to select an image and number of PCA components
    pic = st.slider('Select an image:', 0, 500, 198, key=1)
    n_components = st.slider('Reduce to $n$ dimensions:', 1, 500, 90, key=2)

    # Layout columns for original and eigen images
    col_1, col_2, col_3 = st.columns([1.5, 1.5, 2])

    with col_1:
        # Show the original image
        image_4Darray = X_Bush_uint8[pic]
        fig = plt.figure()
        plt.title('Original Image')
        plt.xlabel('$w$')
        plt.ylabel('$h$')
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
        plt.title('Eigen Image')
        plt.xlabel('$w$')
        plt.ylabel('$h$')
        plt.imshow(X_4D_pca_inv_uint8)
        st.pyplot(fig)