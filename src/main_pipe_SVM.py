
"""
main_pipe_SVM.py
----------------
Script to run a full SVM pipeline with grid search over PCA components, C, and gamma for the LFW face dataset.
Uses the call_pipeline_svm function from pipelines.py and saves results/figures to disk.

Author: Sergej Lembke
Date: 2025-08-19
"""


# --- Local import: SVM pipeline function ---
from pipelines import call_pipeline_svm

# --- Typing imports ---
from typing import List


# --- Hyperparameter grid for grid search ---
C: List[float] = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]  # SVM regularization parameters
GAMMA: List[float] = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]  # SVM kernel coefficients
N_COMPONENTS: List[int] = [60, 80, 100, 120, 140, 160, 180, 200]  # PCA components

# --- Pipeline configuration ---
COLOR: bool = True  # Use color images

# --- Run the SVM pipeline with grid search ---
call_pipeline_svm(
    pipe_name='sca_pca_svm',  # Name for this pipeline run (used in filenames)
    min_faces=100,           # Minimum number of images per person
    color=COLOR,             # Use color images
    n_components=N_COMPONENTS,  # List of PCA components to search
    C=C,                        # List of SVM C values to search
    gamma=GAMMA,                # List of SVM gamma values to search
    ROS=True                    # Use RandomOverSampler for class balancing
)
