import streamlit as st

def app():  
    st.title('Model A - Supervised Learning with Support Vector Machine')
    
    st.subheader('PCA Explained Variance for LFW Dataset')
    st.image(image='plots/pca.png',
            caption='PCA on LFW Dataset',
            width=600)
    st.write('')
    st.write('')
    st.write('')
        
    st.subheader('PCA Explained Variance for LFW Dataset')
    
    st.write('Gridsearch-CV was used to select the appropriate number of components $n$ for PCA reduction, as well as other hyperparameters of the SVM classifier:')

    col_Sa1, col_Sa2 = st.columns([1,1])
    with col_Sa1:
        st.image(image='plots/FIG_bw_100x75_80_sca_pca_svm_2025-05-01_07-06-14.png',
                caption=f'Model A - Black and White: Gridsearch-CV for PCA and SVM',
                width=600)
    with col_Sa2:
        st.image(image='plots/FIG_clr_100x75_80_sca_pca_svm_2025-05-02_22-21-35.png',
                caption=f'Model A - RGB Spectrum: Gridsearch-CV for PCA and SVM',
                width=600)
    
    st.write('The model performs better with RGB images than with black and white images. Therefore, I conducted further analyses only with RGB')