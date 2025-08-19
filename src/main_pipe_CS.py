"""
main_pipe_CS.py
---------------
Script to run a grid search over min_faces and PCA components for the Cosine Similarity pipeline on the LFW face dataset.
Saves results to CSV and prints progress for each combination.

Author: Sergej Lembke
Date: 2025-08-19
"""


# --- Standard library imports ---
import os  # File and directory operations
from datetime import datetime  # For timestamping saved files

# --- Third-party imports ---
import pandas as pd  # DataFrame operations

# --- Local import: Cosine Similarity pipeline function ---
from pipelines import call_pipeline_cs

# --- Typing imports ---
from typing import List


# --- Hyperparameter grid ---
MIN_FACES: List[int] = [140, 100, 60, 40, 30, 20]  # Minimum images per person
N_COMPONENTS: List[int] = [60, 80, 100, 120, 140, 160, 180, 200]  # PCA components

# --- Storage for results ---
val_min_faces: List[int] = []
val_n_components: List[int] = []
val_acc: List[float] = []

counter: int = 0
combinations: int = len(MIN_FACES) * len(N_COMPONENTS)

# --- Grid search over all parameter combinations ---
for min_faces in MIN_FACES:
    for n_components in N_COMPONENTS:
        counter += 1
        print(f'#######################   {counter} / {combinations}   #######################')
        # Run pipeline and collect accuracy
        acc = call_pipeline_cs(
            min_faces=min_faces,
            n_components=n_components,
            color=True,
            ROS=True,
            plot=True
        )
        val_min_faces.append(min_faces)
        val_n_components.append(n_components)
        val_acc.append(acc)

# --- Save results to DataFrame and CSV ---
data = {
    'min_faces': val_min_faces,
    'n_components': val_n_components,
    'accuracy': val_acc
}
df = pd.DataFrame(data=data)

data_dir = os.path.join(os.path.dirname(__file__), "CS_CV_log")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
date_print = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
suffix = f'clr_{date_print}'
data_file = os.path.join(data_dir, f'CS_CV_{suffix}.csv')
df.to_csv(data_file, sep=',')
print(f"Data saved to {data_file}")
