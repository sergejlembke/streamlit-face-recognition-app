import streamlit as st
from sklearn.datasets import fetch_lfw_people

def get_lfw_data(**kwargs):
    """Fetch the LFW dataset with optional kwargs."""
    return fetch_lfw_people(**kwargs)

@st.cache_resource
def get_lfw_data_cached(**kwargs):
    return get_lfw_data(**kwargs)