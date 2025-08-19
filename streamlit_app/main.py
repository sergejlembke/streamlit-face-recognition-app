# --- Third-party imports ---
import streamlit as st

# Set Streamlit page layout to wide
st.set_page_config(layout='wide') 

# Import all Streamlit app pages
import application_page
import eigen_images_page
import introduction_page
import model_a_page
import model_b_page
import model_overview_page
import opencv2_page
import webcam_page

def load_css(file_path: str) -> None:
    """Load a custom CSS file for Streamlit styling."""
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Load custom styles
load_css("styles.css")

# Initialize the session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Welcome' 

# Dictionary mapping page names to their corresponding modules
pages = {
    'Welcome'           : introduction_page, # page 0
    'Eigenimage'        : eigen_images_page, # page 1
    'OpenCV2'           : opencv2_page, # page 2
    'Webcam'            : webcam_page, # page 3
    'Model Overview'    : model_overview_page, # page 4
    'Model A: PCA + SVM': model_a_page, # page 5
    'Model B: PCA + CS' : model_b_page, # page 6
    'Application'       : application_page, # page 7

}

pages_names = list(pages.keys())

# Sidebar navigation for the app
st.sidebar.title('Introduction')
if st.sidebar.button(pages_names[0]):
    st.session_state.page = pages_names[0]
if st.sidebar.button(pages_names[1]):
    st.session_state.page = pages_names[1]

st.sidebar.title('Face Detection')
if st.sidebar.button(pages_names[2]):
    st.session_state.page = pages_names[2]
if st.sidebar.button(pages_names[3]):
    st.session_state.page = pages_names[3]

st.sidebar.title('Face Recognition')
if st.sidebar.button(pages_names[4]):
    st.session_state.page = pages_names[4]
if st.sidebar.button(pages_names[5]):
    st.session_state.page = pages_names[5]
if st.sidebar.button(pages_names[6]):
    st.session_state.page = pages_names[6]
if st.sidebar.button(pages_names[7]):
    st.session_state.page = pages_names[7]

# Render the selected page
pages[st.session_state.page].app()
