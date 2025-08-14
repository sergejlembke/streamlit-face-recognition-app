
import streamlit as st
import os
import cv2
import gc
import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image
from sklearn.decomposition import PCA
from pathlib import Path
from typing import List, Tuple

st.set_page_config(layout='wide') 

import model_overview_page
import eigen_images_page
import introduction_page
import model_a_page
import model_b_page
import comparison_page
import application_page
import webcam_page

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")


if 'page' not in st.session_state:
    st.session_state.page = 'Introduction' 
    # select = 'Introduction'

pages = {
    'Introduction'      : introduction_page,
    'Eigen Images'      : eigen_images_page,
    'Model Overview'    : model_overview_page,
    'Model A: PCA + SVM': model_a_page,
    'Model B: PCA + CS' : model_b_page,
    'Comparison'        : comparison_page,
    'Application'       : application_page,
    'Webcam'            : webcam_page,
    }

pages_names = list(pages.keys())

st.sidebar.title('Introduction')

if st.sidebar.button(pages_names[0]):
    st.session_state.page = pages_names[0]
if st.sidebar.button(pages_names[1]):
    st.session_state.page = pages_names[1]
if st.sidebar.button(pages_names[2]):
    st.session_state.page = pages_names[2]

st.sidebar.title('Face Recognition')

if st.sidebar.button(pages_names[3]):
    st.session_state.page = pages_names[3]
if st.sidebar.button(pages_names[4]):
    st.session_state.page = pages_names[4]
if st.sidebar.button(pages_names[5]):
    st.session_state.page = pages_names[5]
if st.sidebar.button(pages_names[6]):
    st.session_state.page = pages_names[6]
if st.sidebar.button(pages_names[7]):
    st.session_state.page = pages_names[7]

pages[st.session_state.page].app()
