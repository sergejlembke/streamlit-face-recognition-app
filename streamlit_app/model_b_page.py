import streamlit as st

def app():  
    st.title('Model B - No Learning with Cosine Similarity')
       
    st.subheader('Mathematical Background')
    st.write('Cosine similarity is a measure of similarity between two vectors.\nIt calculates the cosine of the angle between the two vectors.')
    st.write('https://en.wikipedia.org/wiki/Cosine_similarity')
    st.write('')
    st.latex(r'Cosine \ Similarity = CS = cos(\theta) = \dfrac{A \cdot B}{||A|| \ ||B||}  \quad , \quad CS \in [-1, \ 1]')
    st.write('')
    st.write(f'$CS = -1$  means a maximally large angle between the data points, i.e., very high dissimilarity.')
    st.write(f'$CS = 0$  means the data points are almost orthogonal, indicating independence.')
    st.write(f'$CS = 1$  means an angle of almost 0° between the data points, i.e., very high similarity.')
    st.write('')
    st.write('')

    st.subheader('Implementation of Cosine Similarity')
    st.write('')
    st.info(f'Step 1:  Calculate CS for each image')
    st.write('For each image from the test dataset X_test, the CS is calculated with each individual image from the training dataset X_train:')
    st.write('(For better illustration, ROS was not applied for the following visualizations)')
    st.write('')
    st.write('')
    
    
    col_Sb1, col_Sb2 = st.columns([1,1])
    with col_Sb1:
        st.image(image='plots/CS_perperson_Schroeder_everyone.png',
                caption=f'CS X_test of Gerhard Schröder with X_train of everyone',
                width=500)
    with col_Sb2:
        st.image(image='plots/CS_perperson_Schroeder_Bush.png',
                caption=f'CS X_test of Gerhard Schröder with X_train of George W. Bush',
                width=500)
    st.write('')
    st.write('')
    st.write('')
    st.write('')

    col_Sc1, col_Sc2, col_Sc3 = st.columns([1,3,1])
    with col_Sc2:
        st.image(image='plots/CS_perperson_Schroeder_Schroeder.png',
                caption=f'CS X_test of Gerhard Schröder with X_train of Gerhard Schröder',
                width=600)
    st.write('')
    st.write('')

    st.divider()
    
    st.write('')

    st.info(f'Step 2:  Calculate CS per image per person')
    st.write('For each image from the test dataset X_test, the CS is calculated for each person from the training dataset X_train.')
    st.write('The CS values for each person are summed up and averaged. The distribution of the CS values is shown in a histogram. The person with the highest CS value is selected as the predicted person for the test image.')
    st.write('Distribution of the CS values for each person is shown in the following plots:')
    st.write('(For better illustration, ROS was not applied for the following visualizations)')
    st.write('')
    st.write('Picture 76 from test dataset:')

    col_Sd1, col_Sd3 = st.columns([2,1])
    
    with col_Sd1:
        st.image(image='plots/CS_pic76.png',
                caption=f'Cosine Similarity for test image 76 with all training images, per person',
                width=1100)
    with col_Sd3:
        st.image(image='plots/CS_dist_76.png',
                caption=f'Wrong Prediction for test image 76. Predicted George W. Bush, but it is Donald Rumsfeld.',
                width=500)
    st.write('')
    
    st.divider()
    
    st.write('')
    st.write('Picture 99 from test dataset:')
    
    col_Se1, col_Se4 = st.columns([2,1])
    with col_Se1:
        st.image(image='plots/CS_pic99.png',
                caption=f'Cosine Similarity for test image 99 with all training images, per person',
                width=1100)
    with col_Se4:
        st.image(image='plots/CS_dist_99.png',
                caption=f'Correct Prediction for test image 99: Predicted Gerhard Schröder.',
                width=500)