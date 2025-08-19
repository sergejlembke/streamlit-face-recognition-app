# --- Third-party imports ---
import streamlit as st


def app() -> None: 
    """
    Streamlit page for Model A: PCA + SVM face recognition.
    This page presents PCA explained variance and model selection results for the LFW dataset.
    """
    # --- Page Title ---
    st.title('Model A: PCA + SVM Face Recognition - Dimensionality Reduction and Hyperparameter Tuning')
    
    # --- PCA explained variance Section ---
    st.subheader('PCA Explained Variance for LFW Dataset')
    st.write(
    """
    The plot below shows the explained variance ratio for each principal component (PC) when applying PCA to the LFW dataset.
    The explained variance ratio indicates how much information (variance) each PC captures from the original data.
    Typically, the first few PCs capture most of the variance, allowing us to reduce dimensionality while retaining most of the important information.
    This plot helps to determine how many components should be kept for effective dimensionality reduction without significant loss of information.
    """
    )

    # Show PCA plot for the LFW dataset
    st.image(
        image='plots/pca.png',
        caption='PCA on LFW Dataset',
        width=600
    )
    
    # Summary of PCA explained variance plot
    st.write(
        """
        The PCA explained variance plot shows how much of the total variance in the LFW dataset is captured by each principal component (PC).
        Cumulative explained variance (CEA) is a measure of how much information from the original data is retained when using the first $n$ PCs.
        For example, at $n=100$ components, the CEA remains above 90%, meaning that more than 90% of the original image information is preserved.
        This allows for significant dimensionality reduction while still retaining the essential features needed for accurate face recognition.
        """
    )

    # Add spacing
    st.write('')
    st.write('')
    st.write('')
    st.write('')

    # --- Explain model selection and hyperparameter tuning Section ---
    st.subheader('Joint Hyperparameter Tuning: PCA Components and SVM Parameters')
    st.write(
    """
    Gridsearch-CV was used to simultaneously select the optimal number of components $n$ for PCA dimensionality reduction
    and the best set of SVM hyperparameters, in order to maximize classification accuracy.
    The following two plots show the results of this joint hyperparameter search, performed separately for black & white (left plot) and RGB images (right plot).

    In each plot, the upper subplots show the explained variance ratio for each principal component (PC), while the lower subplots display the distribution of the 
    classification accuracy (validation) for each parameter setting.
    The vertical lines (error bars) represent the spread (variance) of the accuracy scores across cross-validation folds, providing insight into the stability and 
    reliability of the model's performance for each configuration.
    """
    )

    # Show results of Gridsearch-CV for both black & white and RGB images
    col_1, col_2 = st.columns([1,1])
    with col_1:
        st.image(
            image='plots/FIG_bw_100x75_80_sca_pca_svm_2025-05-01_07-06-14.png',
            caption='Model A - Black and White: Gridsearch-CV for PCA and SVM',
            width=600
        )
    with col_2:
        st.image(
            image='plots/FIG_clr_100x75_80_sca_pca_svm_2025-05-02_22-21-35.png',
            caption='Model A - RGB Spectrum: Gridsearch-CV for PCA and SVM',
            width=600
        )
    
    # Note on model performance
    st.write(
    """
    Comparing the two plots, it is clear that the model achieves higher classification accuracy when using RGB images compared to black & white images.
    The validation accuracy is consistently higher for RGB, and the error bars (variance across folds) are generally smaller, indicating more stable and reliable performance.
    This suggests that the additional color information in RGB images provides valuable features for the SVM classifier, leading to better face recognition results.

    Based on these findings, all further analyses and model development in this project were conducted using RGB images only.
    """
    )