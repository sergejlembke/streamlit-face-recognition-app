"""
main_SVM.py
-----------
Script to train and evaluate an SVM classifier on the LFW face dataset using train_test_svm in pipelines.py.
Generates and displays a confusion matrix and prints classification metrics.

Author: Sergej Lembke
Date: 2025-08-19
"""


# --- Third-party imports ---
import seaborn as sns  # For heatmap visualization
import matplotlib.pyplot as plt  # For plotting
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for matplotlib

# --- Typing imports ---
import numpy as np

# --- Local imports ---
from pipelines import train_test_svm


# --- Configuration ---
min_faces: int = 40  # Minimum number of images per person
color: bool = True  # Use color images
n_components: int = 80  # Number of PCA components
C: float = 1e1  # SVM regularization parameter
gamma: float = 1e-2  # SVM kernel coefficient
ROS: bool = True  # Use RandomOverSampler for class balancing
save: bool = True  # Save trained models


# --- Train and evaluate SVM pipeline ---
acc: float
conf_mat: np.ndarray
class_rep: str
acc, conf_mat, class_rep = train_test_svm(
    min_faces=min_faces,
    color=color,
    n_components=n_components,
    C=C,
    gamma=gamma,
    ROS=ROS,
    save=save
)

print(f"Test accuracy: {acc:.4f}")

# --- Plot confusion matrix ---
plt.figure(figsize=(8, 6))
sns.heatmap(data=conf_mat, annot=True, fmt='d', cmap='OrRd')
plt.suptitle('Confusion Matrix')
plt.title(f'min_faces={min_faces} | n_comp.={n_components} | ROS={ROS} | clr={color}')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# --- Print classification report ---
print(class_rep)


