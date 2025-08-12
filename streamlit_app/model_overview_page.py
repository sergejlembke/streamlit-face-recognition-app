import streamlit as st

def app():    
    st.title('Model Overview')
    
    st.write('Two different models were developed:')
    
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
        st.info('Model A - Supervised Learning')
        st.write('Principal Component Analysis (PCA)')
        st.write('Support Vector Machine (SVM)')
        st.divider() 
        st.info('Version: Accuracy')
        st.write('A1: 96.16%')
        st.write('A2: 90.79%')
        st.write('A3: 85.56%')
        st.write('A4: 74.87%')
        
    with col_S3:
        st.info('Model B - No Learning')
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
    
    st.image(image='plots/numer_of_pictures.png',
            caption='Distribution of images per person in the LFW dataset',
            width=900)
    
    st.write(
        'Since people with many images in the dataset (right edge of the $x$-axis) are the minority, '
        'I used Random Oversampling (ROS) to increase the number of samples for underrepresented classes in the training dataset.'
    )