"""
pipelines.py
------------
Machine learning pipelines for face recognition using the LFW dataset.
Includes SVM and Cosine Similarity pipelines, normalization, and model evaluation utilities.

Author: Sergej Lembke
Date: 2025-08-19
"""


# --- Standard library imports ---
import os  # File and directory operations
import gc  # Garbage collection
from datetime import datetime  # For timestamping saved files

# --- Third-party imports ---
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for matplotlib
import matplotlib.pyplot as plt  # Plotting
import numpy as np  # Numerical operations
import polars as pl  # DataFrame operations (alternative to pandas)
import pandas as pd  # DataFrame operations
import pickle  # Model serialization
import seaborn as sns  # Statistical data visualization
from imblearn.over_sampling import RandomOverSampler  # For class balancing
from sklearn.datasets import fetch_lfw_people  # LFW face dataset
from sklearn.decomposition import PCA  # Principal Component Analysis
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.model_selection import GridSearchCV, train_test_split  # Model selection utilities
from sklearn.metrics.pairwise import cosine_similarity  # Cosine similarity for embeddings
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # Evaluation metrics
from sklearn.pipeline import Pipeline  # ML pipelines
from sklearn.preprocessing import StandardScaler  # Feature scaling


def train_test_svm(
    min_faces: int,
    color: bool,
    n_components: int,
    C: float,
    gamma: float,
    ROS: bool,
    save: bool
) -> Tuple[float, np.ndarray, str]:
    """
    Train and evaluate an SVM classifier on the LFW dataset with optional oversampling.

    Args:
        min_faces: Minimum number of images per person to include.
        h_w: Tuple of (height, width) for image resizing (not used here, but for compatibility).
        color: Whether to use color images.
        n_components: Number of PCA components.
        C: SVM regularization parameter.
        gamma: SVM kernel coefficient.
        train_AUG: Whether to use data augmentation for training (not used here).
        test_AUG: Whether to use data augmentation for testing (not used here).
        ROS: Whether to apply RandomOverSampler to balance classes.
        save: Whether to pickle and save the trained models.

    Returns:
        acc: Accuracy on the test set.
        conf_mat: Confusion matrix.
        class_rep: Classification report as string.
    """
    # Load the LFW people dataset
    lfw = fetch_lfw_people(color=color, resize=0.8, funneled=True, download_if_missing=True)
    X_all = lfw.images  # shape: (n_samples, h, w) if grayscale; or (n_samples, h, w, 3) if RGB
    y_all = lfw.target
    target_names_all = lfw.target_names

    # Count images per label and filter for relevant labels
    unique, counts = np.unique(y_all, return_counts=True)
    labels_count = np.array([(label, count) for label, count in zip(unique, counts)])
    relevant_labels = labels_count[labels_count[:, 1] >= min_faces, 0]

    # Filter images and labels for relevant people
    mask = np.isin(y_all, relevant_labels)
    X = X_all[mask]
    y = y_all[mask]

    # Filter target names
    mask_target_names = np.isin(np.arange(len(target_names_all)), relevant_labels)
    target_names = target_names_all[mask_target_names]
    n_targets = len(target_names)

    print(f"Amount of relevant targets: {n_targets}")
    print(f"Names of relevant targets: {target_names}")
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("Target distribution:")
    for label, count in zip(unique, counts):
        print(f"Target {label}: {count} Pictures")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    del X, X_all, y, y_all

    # Optionally apply RandomOverSampler to balance classes
    if ROS:
        ros = RandomOverSampler(random_state=42)
        n_samples = X_train.shape[0]
        X_train = X_train.reshape(n_samples, -1)
        X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
        del X_train, y_train
        X_train = X_train_res.copy()
        y_train = y_train_res.copy()
        del X_train_res, y_train_res
        print('####### After ROS applied ########')
        unique, counts = np.unique(y_train, return_counts=True)
        print("Target distribution:")
        for label, count in zip(unique, counts):
            print(f"Target {label}: {count} Pictures")

    gc.collect()

    # Normalize pixel values to [0, 1]
    X_train = X_train.astype(np.float64) / 255.0
    X_test = X_test.astype(np.float64) / 255.0

    # Ensure X_train and X_test are 2D (flattened)
    if X_train.ndim > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
    if X_test.ndim > 2:
        X_test = X_test.reshape(X_test.shape[0], -1)

    # Standardize features (zero mean, unit variance)
    sca = StandardScaler()
    X_train = sca.fit_transform(X_train)
    X_test = sca.transform(X_test)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Train SVM classifier
    svm = SVC(C=C, gamma=gamma, class_weight='balanced', kernel='rbf')
    svm.fit(X_train_pca, y_train)
    y_pred = svm.predict(X_test_pca)

    # Optionally save models
    if save:
        date_print = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename_raw = f'{min_faces}_C={C}_gamma={gamma}_PCAn={n_components}_{date_print}.sav'
        pickle.dump(svm, open(f'SVM_{filename_raw}', 'wb'))
        print('SVM PICKLED')
        pickle.dump(sca, open(f'SCA_{filename_raw}', 'wb'))
        print('SCA PICKLED')
        pickle.dump(pca, open(f'PCA_{filename_raw}', 'wb'))
        print('PCA PICKLED')

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    class_rep = classification_report(y_test, y_pred)
    return acc, conf_mat, class_rep



def call_pipeline_svm(
    pipe_name: str,
    min_faces: int,
    color: bool,
    n_components: list[int],
    C: list[float],
    gamma: list[float],
    ROS: bool
) -> None:
    """
    Run a full SVM pipeline with grid search and PCA for the LFW dataset.

    Args:
        pipe_name: Name for the pipeline run (used in filenames).
        min_faces: Minimum number of images per person to include.
        h_w: Tuple of (height, width) for image resizing (not used here).
        color: Whether to use color images.
        n_components: List of PCA components to grid search.
        C: List of SVM regularization parameters to grid search.
        gamma: List of SVM kernel coefficients to grid search.
        train_AUG: Whether to use data augmentation for training (not used here).
        test_AUG: Whether to use data augmentation for testing (not used here).
        ROS: Whether to apply RandomOverSampler to balance classes.

    Returns:
        None. Saves results and figures to disk.
    """
    # Load the LFW people dataset
    lfw = fetch_lfw_people(color=color, resize=0.8, funneled=True, download_if_missing=True)
    X_all = lfw.data
    y_all = lfw.target
    target_names_all = lfw.target_names

    h = lfw.images.shape[1]
    w = lfw.images.shape[2]

    # Count images per label and filter for relevant labels
    unique, counts = np.unique(y_all, return_counts=True)
    labels_count = np.array([(label, count) for label, count in zip(unique, counts)])
    relevant_labels = labels_count[labels_count[:, 1] >= min_faces, 0]

    # Filter images and labels for relevant people
    mask = np.isin(y_all, relevant_labels)
    X = X_all[mask]
    y = y_all[mask]

    # Filter target names
    mask_target_names = np.isin(np.arange(len(target_names_all)), relevant_labels)
    target_names = target_names_all[mask_target_names]
    n_targets = len(target_names)

    print(f"Amount of relevant targets: {n_targets}")
    print(f"Names of relevant targets: {target_names}")
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("Target distribution:")
    for label, count in zip(unique, counts):
        print(f"Target {label}: {count} Pictures")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    del X, X_all, y, y_all

    # Optionally apply RandomOverSampler to balance classes
    if ROS:
        ros = RandomOverSampler(random_state=42)
        X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
        del X_train, y_train
        X_train = X_train_res
        y_train = y_train_res
        del X_train_res, y_train_res

    gc.collect()

    # Normalize pixel values to [0, 1]
    X_train = X_train.astype(np.float64) / 255.0
    X_test = X_test.astype(np.float64) / 255.0

    # Ensure X_train and X_test are 2D (flattened)
    if X_train.ndim > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
    if X_test.ndim > 2:
        X_test = X_test.reshape(X_test.shape[0], -1)

    # Build pipeline: StandardScaler -> PCA -> SVM
    scaler = StandardScaler()
    pca = PCA(whiten=True)
    svm = SVC(class_weight='balanced', kernel='rbf')
    pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("svm", svm)])

    # Grid search over PCA n_components, SVM C, and gamma
    param_grid = {
        "pca__n_components": n_components,
        'svm__C': C,
        'svm__gamma': gamma,
    }
    search = GridSearchCV(pipe, param_grid, n_jobs=2, verbose=2)
    search.fit(X_train, y_train).score(X_test, y_test)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)

    # Save grid search results to CSV
    df_gridsearch = pd.DataFrame(search.cv_results_)
    if color:
        data_dir = os.path.join(os.path.dirname(__file__), "clr/CV_log")
    else:
        data_dir = os.path.join(os.path.dirname(__file__), "clr/CV_log")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    date_print = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if color:
        suffix = f'clr_{h}x{w}_{min_faces}_{pipe_name}_{date_print}'
    else:
        suffix = f'bw_{h}x{w}_{min_faces}_{pipe_name}_{date_print}'
    data_file = os.path.join(data_dir, f'CV_{suffix}.csv')
    df_gridsearch.to_csv(data_file, sep=',')
    print(f"Data saved to {data_file}")

    # Plot the PCA spectrum and save figure
    pca.fit(X_train)
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(10, 7))
    if color:
        plt.suptitle(f"LFW_clr_{h}x{w}_{min_faces}_{pipe_name}", fontsize=14)
    else:
        plt.suptitle(f"LFW_bw_{h}x{w}_{min_faces}_{pipe_name}", fontsize=14)
    plt.title(f"Best parameter (CV score={search.best_score_:.3f})\n{search.best_params_}", fontsize=10)
    ax0.plot(np.cumsum(pca.explained_variance_ratio_))
    ax0.set_ylabel("PCA cumulative explained variance")
    ax0.axvline(
        search.best_estimator_.named_steps["pca"].n_components,
        linestyle=":",
        label="n_components chosen",
        c='r',
    )
    ax0.legend(prop=dict(size=10))
    # For each number of components, find the best classifier results
    components_col = "param_pca__n_components"
    is_max_test_score = pl.col("mean_test_score") == pl.col("mean_test_score").max()
    best_clfs = (
        pl.LazyFrame(search.cv_results_, strict=False)
        .filter(is_max_test_score.over(components_col))
        .unique(components_col)
        .sort(components_col)
        .collect()
    )
    ax1.errorbar(
        best_clfs[components_col],
        best_clfs["mean_test_score"],
        yerr=best_clfs["std_test_score"],
    )
    ax1.set_ylabel("Classification accuracy (val)")
    ax1.set_xlabel("n_components")
    plt.xlim(-1, max(n_components)  + 100)
    plt.tight_layout()
    # Save figure to file
    if color:
        fig_dir = os.path.join(os.path.dirname(__file__), "clr/FIG_log")
    else:
        fig_dir = os.path.join(os.path.dirname(__file__), "bw/FIG_log")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig_file = os.path.join(fig_dir, f'FIG_{suffix}.png')
    fig.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {fig_file}")
    plt.close(fig)
    del fig
    return







def call_pipeline_cs(
    min_faces: int,
    n_components: int,
    color: bool,
    ROS: bool,
    plot: bool
) -> float:
    """
    Run a Cosine Similarity pipeline for face recognition on the LFW dataset.

    Args:
        h_w: Tuple of (height, width) for image resizing (not used here).
        min_faces: Minimum number of images per person to include.
        n_components: Number of PCA components.
        color: Whether to use color images.
        train_AUG: Whether to use data augmentation for training (not used here).
        test_AUG: Whether to use data augmentation for testing (not used here).
        ROS: Whether to apply RandomOverSampler to balance classes.
        plot: Whether to plot the confusion matrix.

    Returns:
        Accuracy score on the test set.
    """
    # Load the LFW people dataset
    lfw = fetch_lfw_people(color=color, resize=0.8, funneled=True, download_if_missing=True)
    X_all = lfw.images
    y_all = lfw.target
    target_names_all = lfw.target_names

    # Count images per label and filter for relevant labels
    unique, counts = np.unique(y_all, return_counts=True)
    labels_count = np.array([(label, count) for label, count in zip(unique, counts)])
    relevant_labels = labels_count[labels_count[:, 1] >= min_faces, 0]

    # Filter images and labels for relevant people
    mask = np.isin(y_all, relevant_labels)
    X = X_all[mask]
    y = y_all[mask]

    # Filter target names
    mask_target_names = np.isin(np.arange(len(target_names_all)), relevant_labels)
    target_names = target_names_all[mask_target_names]
    n_targets = len(target_names)

    print(f"Amount of relevant targets: {n_targets}")
    print(f"Names of relevant targets: {target_names}")
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("Target distribution:")
    for label, count in zip(unique, counts):
        print(f"Target {label}: {count} Pictures")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    del X, X_all, y, y_all

    # Optionally apply RandomOverSampler to balance classes
    if ROS:
        ros = RandomOverSampler(random_state=42)
        X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
        del X_train, y_train
        X_train = X_train_res
        y_train = y_train_res
        del X_train_res, y_train_res

    gc.collect()

    # Normalize pixel values to [0, 1]
    X_train = X_train.astype(np.float64) / 255.0
    X_test = X_test.astype(np.float64) / 255.0

    # Ensure X_train and X_test are 2D (flattened)
    if X_train.ndim > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
    if X_test.ndim > 2:
        X_test = X_test.reshape(X_test.shape[0], -1)

    # Standardize features (zero mean, unit variance)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Cosine similarity classification
    cs_sum_test = np.empty(shape=(X_test_pca.shape[0], n_targets))
    for picture in range(0, X_test_pca.shape[0]):
        cs_picture_sum = np.empty(shape=(0, n_targets))
        for person in relevant_labels:
            # Get all training embeddings for this person
            person_data = X_train_pca[y_train == person]
            if person_data.shape[0] == 0:
                cs_picture_sum = np.append(cs_picture_sum, 0)  # No contribution if no data
                continue
            # Compute cosine similarity and sum
            cs_picture_person = cosine_similarity(person_data, X_test_pca[picture].reshape(1, -1)) ** 3
            cs_picture_sum = np.append(cs_picture_sum, cs_picture_person.sum())
        cs_sum_test[picture] = cs_picture_sum

    # Assign predicted label as the one with max similarity
    cs_pred = []
    for pic_id in range(cs_sum_test.shape[0]):
        for idx, value in np.ndenumerate(cs_sum_test[pic_id]):
            if value == max(cs_sum_test[pic_id]):
                cs_pred.append(relevant_labels[idx[0]])

    # Optionally plot confusion matrix
    if plot:
        sns.heatmap(confusion_matrix(y_test, cs_pred), annot=True)
        plt.title(f'min_faces={min_faces} | n_comp.={n_components} | ROS={ROS} | clr={color}')
        plt.suptitle('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()

    return accuracy_score(y_test, cs_pred)

