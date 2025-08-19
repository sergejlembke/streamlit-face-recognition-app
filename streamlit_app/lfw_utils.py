# --- Third-party imports ---
import streamlit as st
from sklearn.datasets import fetch_lfw_people

@st.cache_resource
def get_lfw_data_cached(**kwargs):
    """Fetch the LFW dataset with optional kwargs."""
    return fetch_lfw_people(**kwargs)