import sys
import os
import gc
import pickle

import numpy as np

from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import fetch_lfw_people

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lfw_utils import get_lfw_data_cached
lfw = get_lfw_data_cached(color=True, resize=0.8, funneled=True, download_if_missing=True)



def apply_model_A(selected_model: str, crop: np.ndarray) -> tuple:
    directory = 'model_A/'
    
    if 'A1' in selected_model:
        filename_raw = "140_C=1.0_gamma=0.01_PCAn=70_2025-05-09_06-14-04.sav"
    elif 'A2' in selected_model:
        filename_raw = "100_C=1.0_gamma=0.01_PCAn=70_2025-05-09_06-15-38.sav"
    elif 'A3' in selected_model:
        filename_raw = "60_C=1.0_gamma=0.01_PCAn=80_2025-05-09_07-11-12.sav"
    elif 'A4' in selected_model:
        filename_raw = '40_C=10.0_gamma=0.01_PCAn=80_2025-05-09_08-30-10.sav'
        
    # load the scaler, pca and svm
    sca = pickle.load(open(f'{directory}SCA_{filename_raw}', 'rb'))
    pca = pickle.load(open(f'{directory}PCA_{filename_raw}', 'rb'))
    svm = pickle.load(open(f'{directory}SVM_{filename_raw}', 'rb'))

    X_2D = crop.reshape(1,-1).astype(np.float64) / 255.0
    X_2D = sca.transform(X_2D)
    X_2D = pca.transform(X_2D)
    
    return pca, sca, svm, X_2D





def preprocess_cs(selected_model: str, crop: np.ndarray) -> tuple:
        
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
    
    lfw = fetch_lfw_people(color=True, resize=0.8, funneled=True, download_if_missing=True)

    X_all = lfw.images
    y_all = lfw.target
    target_names_all = lfw.target_names

    unique, counts = np.unique(y_all, return_counts=True)
    labels_count = np.array([(label, count) for label, count in zip(unique, counts)])

    # Extract the  relevant labels based on min_faces
    relevant_labels = labels_count[labels_count[:, 1] >= min_faces, 0]

    # Filter X_all and y_all based on the relevant labels
    mask = np.isin(y_all, relevant_labels)
    X = X_all[mask]
    y = y_all[mask]

    # Filter target_names_all based on the relevant labels
    mask_target_names = np.isin(np.arange(len(target_names_all)), relevant_labels)
    target_names = target_names_all[mask_target_names]

    n_targets = len(target_names)

    # Split data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    del X, X_all, X_test, y_test, y

    ros = RandomOverSampler(random_state=42)
    n_samples = X_train.shape[0]
    X_train_flat = X_train.reshape(n_samples, -1)
    X_train_res, y_train_res = ros.fit_resample(X_train_flat, y_train)

    # X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
    del X_train, y_train

    X_train = X_train_res
    y_train = y_train_res
    del X_train_res, y_train_res

    gc.collect()

    X_train = X_train.astype(np.float64) / 255.0

    # Standardize the data
    sca = StandardScaler()
    X_train = sca.fit_transform(X_train)
                                
    # PCA on training data, then dimension reduction on training and test data
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
                                                                                
    X_2D = crop.reshape(1,-1).astype(np.float64) / 255.0
    X_2D = sca.transform(X_2D)
    X_2D = pca.transform(X_2D)
    
    return  pca, sca, X_2D, X_train_pca, y_train, y_all, relevant_labels, target_names_all, n_targets



def calculate_cs(X_2D: np.ndarray, X_train_pca: np.ndarray, y_train, y_all: np.ndarray, relevant_labels: np.ndarray, n_targets: int) -> np.ndarray:
    
    cs_sum_test = np.empty(shape=(X_2D.shape[0], n_targets))

    for picture in range(0, X_2D.shape[0]):
        cs_picture_sum = np.empty(shape=(0, n_targets))
        
        for person in relevant_labels:
            # Filter für die aktuelle Person
            person_data = X_train_pca[y_train == person]
            
            # Überprüfen, ob Daten für die Person vorhanden sind
            if person_data.shape[0] == 0:
                cs_picture_sum = np.append(cs_picture_sum, 0.0)  # Kein Beitrag zur Ähnlichkeit
                continue
            
            # Berechnung der Cosine Similarity
            cs_picture_person = cosine_similarity(person_data,
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