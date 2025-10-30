
"""
create_CS_conf_matrix.py
------------------------
Script to generate and plot a confusion matrix for the Cosine Similarity pipeline on the LFW face dataset.
Runs the pipeline for a single parameter combination and displays the confusion matrix.

Author: Sergej Lembke
Date: 2025-08-19
"""


# --- Local import: Cosine Similarity pipeline function ---
from pipelines import call_pipeline_cs


# --- Parameter configuration ---
MIN_FACES: list[int] = [40]      # Minimum images per person
N_COMPONENTS: list[int] = [450]  # Number of PCA components

counter: int = 0
combinations: int = len(MIN_FACES) * len(N_COMPONENTS)

# --- Run pipeline and plot confusion matrix ---
for min_faces in MIN_FACES:
    for n_components in N_COMPONENTS:
        counter += 1
        print(f'#######################   {counter} / {combinations}   #######################')
        # Run pipeline and plot confusion matrix
        acc = call_pipeline_cs(
            min_faces=min_faces,
            n_components=n_components,
            color=True,
            ROS=True,
            plot=True
        )