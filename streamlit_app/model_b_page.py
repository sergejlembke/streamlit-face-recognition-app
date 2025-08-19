# --- Standard library imports ---
import os

# --- Third-party imports ---
import streamlit as st


def app() -> None:
    """
    Streamlit page for Model B: Cosine Similarity Face Recognition (Non-Learning Approach).
    This page explains and visualizes a baseline face recognition method that does not use machine learning,
    but instead compares feature vectors using cosine similarity.
    """
    # --- Page Title ---
    st.title('Model B: Cosine Similarity Face Recognition - Non-Learning Baseline')

    # --- Mathematical Background Section ---
    st.subheader('Mathematical Background')
    # Brief intro and link to cosine similarity
    st.write(f'[Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) $(CS)$ is a measure of similarity between two vectors $A$ and $B$. It calculates the cosine of the angle $\\theta$ between the two vectors, and is widely used in high-dimensional spaces such as image recognition.')
    st.write('')
    # Show the mathematical formula
    st.latex(r'CS = \cos(\theta) = \dfrac{A \cdot B}{||A|| \, ||B||}  \quad , \quad CS \in [-1, \, 1]')
    st.write('')
    # Geometric interpretation
    st.write('**Geometric Interpretation:**')
    st.write('Cosine similarity measures the cosine of the angle $\\theta$ between two vectors $A$ and $B$ in a multi-dimensional space. If the vectors point in the same direction, the angle is 0° and the cosine similarity is 1. If they are orthogonal (90°), the similarity is 0. If they point in opposite directions (180°), the similarity is -1.')
    st.write('')
    # Why cosine similarity is used for face recognition
    st.write('**Why use Cosine Similarity for Face Recognition?**')
    st.write('In face recognition, each image is represented as a high-dimensional vector. Cosine similarity focuses on the direction of these vectors, not their magnitude. This makes it robust to differences in lighting or scale, as it compares the pattern or structure of the faces rather than their absolute values.')
    st.write('')
    # Example calculation for educational clarity
    st.write('**Example Calculation:**')
    st.write('Suppose $A = [1, 2]$ and $B = [2, 3]$.')
    st.latex(r'A \cdot B = 1 \times 2 + 2 \times 3 = 2 + 6 = 8')
    st.latex(r'||A|| = \sqrt{1^2 + 2^2} = \sqrt{1 + 4} = \sqrt{5}')
    st.latex(r'||B|| = \sqrt{2^2 + 3^2} = \sqrt{4 + 9} = \sqrt{13}')
    st.latex(r'CS = \frac{8}{\sqrt{5} \times \sqrt{13}} \approx 0.992')
    st.write('This value close to 1 indicates that the vectors are very similar (small angle between them).')
    st.write('')
    st.write('')

    # --- Implementation Section ---
    st.subheader('Implementation of Cosine Similarity')
    st.write('')

    # --- Step 1: Calculate CS for each image ---
    col_S1, col_S2, = st.columns([1, 1]) 
    with col_S1:
        st.info(f'**Step 1:  Calculate $CS$ for each image in $X_{{test}}$ with each image in $X_{{train}}$**')
    # Explain the process of comparing each test image to all training images
    st.write('In this step, each test image from $X_{{test}}$ is compared with every training image in $X_{{train}}$ by calculating the $CS$ between their feature vectors. This results in a set of $CS$ values for each test image, one for each training image. The higher the $CS$, the more similar the test image is to the corresponding training image.')
    st.write('')
    # Describe the plots for Step 1
    st.write('The following plots illustrate this process for images of Gerhard Schröder:')
    st.write('- **Left plot:** Shows the cosine similarity between the images from Gerhard Schröder from $X_{{test}}$ and all training images of every person in the training set  $X_{{train}}$. This gives an overview of how similar the test image is to the entire training set.')
    st.write('- **Right plot:** Focuses on the cosine similarity between the test images from Gerhard Schröder and only the training images of George W. Bush, allowing for a more detailed comparison with a specific individual.')
    st.write('')
    st.write('A higher $CS$ value indicates greater similarity. Ideally, the test image should have the highest $CS$ values with training images of the same person.')
    st.write('')

    # --- Plots for Step 1 ---
    col_1, col_2 = st.columns([1,1])
    with col_1:
        st.image(
            image=os.path.join(os.path.dirname(__file__), 'plots', 'CS_perperson_Schroeder_everyone.png'),
            caption=f'$CS$ for X_test of Gerhard Schröder with X_train of everyone',
            width=500
        )
    with col_2:
        st.image(
            image=os.path.join(os.path.dirname(__file__), 'plots', 'CS_perperson_Schroeder_Bush.png'),
            caption=f'$CS$ for X_test of Gerhard Schröder with X_train of George W. Bush',
            width=500
        )
    st.write('')
    st.write('')
    st.write('')
    st.write('')

    # --- Center plot: CS with same person ---
    st.write('The following plot displays the $CS$ for test images of Gerhard Schröder on all training images of Gerhard Schröder. As expected, a lot of $CS$ values are high, indicating strong similarities. Though there are many pictures with lower $CS$ values, around 0, which indicates that the model is not perfect and can make mistakes in identifying faces.')
    st.write('')
    col_1, col_2, col_3 = st.columns([1,3,1])
    with col_2:
        st.image(
            image=os.path.join(os.path.dirname(__file__), 'plots', 'CS_perperson_Schroeder_Schroeder.png'),
            caption=f'$CS$ for X_test of Gerhard Schröder with X_train of Gerhard Schröder',
            width=600
        )
    st.write('')
    st.write('')
    st.write('')

    # --- Step 2: Aggregate CS per image per person and make predictions ---
    st.divider()
    st.write('')
    col_S1, col_S2, = st.columns([1, 1]) 
    with col_S1:
        st.info(f'**Step 2:  Aggregate $CS$ per image per person and make predictions**')
    # Explain the aggregation and prediction process
    st.write('In this step, we move from comparing individual images to making a prediction for each test image. For every test image in $X_{{test}}$, we calculate the $CS$ with all training images in $X_{{train}}$, grouped by person. For each person, we sum and average the $CS$ values between the test image and all of their training images. This results in one average $CS$ value per person for each test image.')
    st.write('')
    st.write('The person with the highest average $CS$ value is selected as the predicted identity for the test image. This approach leverages the idea that a test image should be most similar, on average, to images of the same person in the training set.')
    st.write('')
    # Describe the plots for Step 2
    st.write('The following plots illustrate this process for two example test images:')
    st.write('- **Left plot:** Shows the cosine similarity between a specific test image and all training images, grouped by person. Each subplot represents a person.')
    st.write('- **Right plot:** Displays the aggregation of all $CS$ values for the test image per person, providing insight into how well the test image matches the training set overall. The histogram can reveal whether the prediction is clear-cut or ambiguous.')
    st.write('')
    st.write('**Image 76 from test dataset:**')
    st.write('In this example, the right plot shows that the highest aggregated $CS$ value is for George W. Bush, so the model predicts this as the identity (labeled as the red bar). However, the true identity is Donald Rumsfeld, indicated as the green bar. This demonstrates a case where the model makes an incorrect prediction, possibly due to similarities in facial features or limitations in the dataset.')
    # --- Plots for Step 2, Example 1 ---
    col_1, col_2 = st.columns([2,1])
    with col_1:
        st.image(
            image=os.path.join(os.path.dirname(__file__), 'plots', 'CS_pic76.png'),
            caption=f'$CS$ for test image 76 with all training images, grouped by person',
            width=900
        )
    with col_2:
        st.image(
            image=os.path.join(os.path.dirname(__file__), 'plots', 'CS_dist_76.png'),
            caption=f'Aggregated $CS$ values for test image 76. Wrong prediction: Predicted George W. Bush, actual is Donald Rumsfeld.',
            width=500
        )
    st.write('')
    st.divider()
    st.write('')
    st.write('**Image 99 from test dataset:**')
    st.write('Here, the right plot shows that the highest aggregated $CS$ value is for Gerhard Schröder, which matches the true identity (green bar). This example demonstrates a successful identification by the model.')
    # --- Plots for Step 2, Example 2 ---
    col_1, col_2 = st.columns([2,1])
    with col_1:
        st.image(
            image=os.path.join(os.path.dirname(__file__), 'plots', 'CS_pic99.png'),
            caption=f'$CS$ for test image 99 with all training images, grouped by person',
            width=900
        )
    with col_2:
        st.image(
            image=os.path.join(os.path.dirname(__file__), 'plots', 'CS_dist_99.png'),
            caption=f'Aggregated $CS$ values for test image 99. Correct prediction: Gerhard Schröder.',
            width=500
        )