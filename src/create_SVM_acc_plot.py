
"""
create_SVM_acc_plot.py
----------------------
Script to run a grid search over SVM C and gamma parameters for the LFW face dataset.
Plots a heatmap of accuracy scores for each parameter combination and saves the figure.

Author: Sergej Lembke
Date: 2025-08-19
"""


# --- Local import: SVM pipeline function ---
from pipelines import train_test_svm

# --- Third-party imports ---
import seaborn as sns  # For heatmap visualization
import matplotlib.pyplot as plt  # For plotting
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for matplotlib
import numpy as np  # Numerical operations

# --- Typing imports ---
from typing import List


# --- Parameter configuration ---
min_faces: int = 40  # Minimum number of images per person
color: bool = True  # Use color images
n_components: int = 90  # Number of PCA components
C: List[float] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]  # SVM C values
GAMMA: List[float] = [1e-4, 1e-3, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]  # SVM gamma values
ROS: bool = True  # Use RandomOverSampler for class balancing
save: bool = False  # Do not save models

# --- Storage for accuracy scores ---
scores: np.ndarray = np.ndarray(shape=(len(C), len(GAMMA)))

# --- Grid search over C and gamma ---
for i, c in enumerate(C):
    scores_C = []
    for gamma in GAMMA:
        acc, conf_mat, class_rep = train_test_svm(
            min_faces=min_faces,
            color=color,
            n_components=n_components,
            C=c,
            gamma=gamma,
            ROS=ROS,
            save=save
        )
        scores_C.append(acc)
    scores[i] = scores_C

# --- Plot heatmap of accuracy scores ---
plt.figure(figsize=(8, 6))
sns.heatmap(
    data=scores * 100,
    annot=True,
    yticklabels=C,
    xticklabels=GAMMA,
    cmap='rocket',
    vmin=0,
    vmax=100,
    fmt='.3g'
)
plt.suptitle('Accuracy')
plt.title(f'min_faces={min_faces} | n_comp.={n_components} | ROS={ROS} | clr={color}')
plt.xlabel('gamma')
plt.ylabel('C')
plt.tight_layout()
plt.show()

# --- Save figure to file ---
plt.savefig(f'SVM/scores_minFa={min_faces}_n={n_components}_ROS={ROS}_clr={color}.png')

