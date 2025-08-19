# --- Standard library imports ---
import gc
import pickle

# --- Third-party imports ---
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# --- Local imports ---
from lfw_utils import get_lfw_data_cached


# Fetch the LFW dataset (cached)
lfw = get_lfw_data_cached(color=True, resize=0.8, funneled=True, download_if_missing=True)

def apply_model_A(selected_model: str, crop: np.ndarray) -> tuple:
    """
    Load the appropriate SVM, PCA, and scaler models for the selected SVM model,
    preprocess the cropped face, and return the fitted objects and transformed face.
    """
    directory = 'model_A/'
    
    # Select model file based on user selection
    if 'A1' in selected_model:
        filename_raw = "140_C=1.0_gamma=0.01_PCAn=70_2025-05-09_06-14-04.sav"
    elif 'A2' in selected_model:
        filename_raw = "100_C=1.0_gamma=0.01_PCAn=70_2025-05-09_06-15-38.sav"
    elif 'A3' in selected_model:
        filename_raw = "60_C=1.0_gamma=0.01_PCAn=80_2025-05-09_07-11-12.sav"
    elif 'A4' in selected_model:
        filename_raw = '40_C=10.0_gamma=0.01_PCAn=80_2025-05-09_08-30-10.sav'
        
    # Load the scaler, PCA, and SVM models
    sca = pickle.load(open(f'{directory}SCA_{filename_raw}', 'rb'))
    pca = pickle.load(open(f'{directory}PCA_{filename_raw}', 'rb'))
    svm = pickle.load(open(f'{directory}SVM_{filename_raw}', 'rb'))

    # Preprocess the cropped face for prediction
    X_2D = crop.reshape(1, -1).astype(np.float64) / 255.0
    X_2D = sca.transform(X_2D)
    X_2D = pca.transform(X_2D)
    
    return pca, sca, svm, X_2D

def preprocess_cs(selected_model: str, crop: np.ndarray) -> tuple:
    """
    Preprocess the data for the cosine similarity model:
    - Selects relevant labels based on the number of faces per person
    - Balances the dataset
    - Standardizes and applies PCA
    - Prepares the cropped face for recognition
    Returns all necessary objects and arrays for recognition.
    """
    # Set PCA and minimum faces per person based on model selection
    if 'B1' in selected_model:
        n_components = 160
        min_faces = 140
    elif 'B2' in selected_model:
        n_components = 200
        min_faces = 100
    elif 'B3' in selected_model:
        n_components = 360
        min_faces = 60
    elif 'B4' in selected_model:
        n_components = 450
        min_faces = 40
    
    X_all = lfw.images
    y_all = lfw.target
    target_names_all = lfw.target_names

    # Count the number of images per label
    unique, counts = np.unique(y_all, return_counts=True)
    labels_count = np.array([(label, count) for label, count in zip(unique, counts)])

    # Extract relevant labels based on min_faces
    relevant_labels = labels_count[labels_count[:, 1] >= min_faces, 0]

    # Filter X_all and y_all based on the relevant labels
    mask = np.isin(y_all, relevant_labels)
    X = X_all[mask]
    y = y_all[mask]

    # Filter target_names_all based on the relevant labels
    mask_target_names = np.isin(np.arange(len(target_names_all)), relevant_labels)
    target_names = target_names_all[mask_target_names]

    n_targets = len(target_names)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Free memory
    del X, X_all, X_test, y_test, y

    # Balance the training data using RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    n_samples = X_train.shape[0]
    X_train_flat = X_train.reshape(n_samples, -1)
    X_train_res, y_train_res = ros.fit_resample(X_train_flat, y_train)

    del X_train, y_train

    X_train = X_train_res
    y_train = y_train_res
    del X_train_res, y_train_res

    gc.collect()

    # Normalize pixel values
    X_train = X_train.astype(np.float64) / 255.0

    # Standardize the data
    sca = StandardScaler()
    X_train = sca.fit_transform(X_train)
                                
    # PCA on training data
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
                                                                                
    # Preprocess the cropped face for recognition
    X_2D = crop.reshape(1, -1).astype(np.float64) / 255.0
    X_2D = sca.transform(X_2D)
    X_2D = pca.transform(X_2D)
    
    return  pca, sca, X_2D, X_train_pca, y_train, y_all, relevant_labels, target_names_all, n_targets

def calculate_cs(
    X_2D: np.ndarray,
    X_train_pca: np.ndarray,
    y_train,
    y_all: np.ndarray,
    relevant_labels: np.ndarray,
    n_targets: int
) -> np.ndarray:
    """
    Calculate cosine similarity between the input face and all faces in the training set,
    sum similarities for each person, and return a mask for the most similar identity.
    """
    cs_sum_test = np.empty(shape=(X_2D.shape[0], n_targets))

    for picture in range(0, X_2D.shape[0]):
        cs_picture_sum = np.empty(shape=(0, n_targets))
        
        for person in relevant_labels:
            # Filter for the current person
            person_data = X_train_pca[y_train == person]
            
            # Check if data for the person exists
            if person_data.shape[0] == 0:
                cs_picture_sum = np.append(cs_picture_sum, 0.0)  # No contribution to similarity
                continue
            
            # Calculate cosine similarity and raise to the power of 3
            cs_picture_person = cosine_similarity(
                person_data,
                X_2D[picture].reshape(1, -1),
            ) ** 3
            
            cs_picture_sum = np.append(cs_picture_sum, cs_picture_person.sum())

        cs_sum_test[picture] = cs_picture_sum

    cs_pred = []
    for pic_id in range(cs_sum_test.shape[0]):
        for idx, value in np.ndenumerate(cs_sum_test[pic_id]):
            if value == max(cs_sum_test[pic_id]):        
                cs_pred.append(relevant_labels[idx[0]])

    mask = np.isin(np.unique(y_all), cs_pred)
    
    return mask