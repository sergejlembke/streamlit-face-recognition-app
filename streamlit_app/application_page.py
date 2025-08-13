import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
import face_detection
import face_recognition


def app():
    st.title('Zusammenführen von Face Detection & Face Recognition')
    st.write('')
    st.write('')
    
    col_SL11, col_SR11 = st.columns([1,1])
    
    with col_SL11:
        selected_model = st.radio("Modell für Prediction",
                                [':rainbow[A1: PCA + SVM [96,15 % Accuracy]]', ':rainbow[A2: PCA + SVM [90,79 % Accuracy]]', ':rainbow[A3: PCA + SVM [85,56 & Accuracy]]', ':rainbow[A4: PCA + SVM [74,87 & Accuracy]]', ':rainbow[B1: PCA + CS [96,15 % Accuracy]]', ':rainbow[B2: PCA + CS [90,79 % Accuracy]]', ':rainbow[B3: PCA + CS [87,78 % Accuracy]]', ':rainbow[B4: PCA + CS [77,54 % Accuracy]]'],
                                captions=["3 Personen  [Colin Powell, George W Bush, Tony Blair]",
                                            "5 Personen  [Colin Powell, Donald Rumsfeld, George W Bush, Gerhard Schroeder, Tony Blair]",
                                            "8 Personen  [Ariel Sharon, Colin Powell, Donald Rumsfeld, George W Bush, Gerhard Schroeder, Hugo Chavez, Junichiro Koizumi, Tony Blair]",
                                            "19 Personen [Ariel Sharon, Arnold Schwarzenegger, Colin Powell, Donald Rumsfeld, George W Bush, Gerhard Schroeder, Gloria Macapagal Arroyo, Hugo Chavez, Jacques Chirac, Jean Chretien, Jennifer Capriati, John Ashcroft, Junichiro Koizumi, Laura Bush, Lleyton Hewitt, Luiz Inacio Lula da Silva, Serena Williams, Tony Blair, Vladimir Putin]",
                                            "3 Personen  [Colin Powell, George W Bush, Tony Blair]",
                                            "5 Personen  [Colin Powell, Donald Rumsfeld, George W Bush, Gerhard Schroeder, Tony Blair]",
                                            "8 Personen  [Ariel Sharon, Colin Powell, Donald Rumsfeld, George W Bush, Gerhard Schroeder, Hugo Chavez, Junichiro Koizumi, Tony Blair]",
                                            "19 Personen [Ariel Sharon, Arnold Schwarzenegger, Colin Powell, Donald Rumsfeld, George W Bush, Gerhard Schroeder, Gloria Macapagal Arroyo, Hugo Chavez, Jacques Chirac, Jean Chretien, Jennifer Capriati, John Ashcroft, Junichiro Koizumi, Laura Bush, Lleyton Hewitt, Luiz Inacio Lula da Silva, Serena Williams, Tony Blair, Vladimir Putin]"
                                            ]
                                ) 
    with col_SR11:
        if 'A1' in selected_model:
            filename_CM = "plots/CM_A_140.png"
        elif 'A2' in selected_model:
            filename_CM = "plots/CM_A_100.png"
        elif 'A3' in selected_model:
            filename_CM = "plots/CM_A_60.png"
        elif 'A4' in selected_model:
            filename_CM = "plots/CM_A_40.png"
        elif 'B1' in selected_model:
            filename_CM = "plots/CM_B_140.png"
        elif 'B2' in selected_model:
            filename_CM = "plots/CM_B_100.png"
        elif 'B3' in selected_model:
            filename_CM = "plots/CM_B_60.png"
        elif 'B4' in selected_model:
            filename_CM = "plots/CM_B_40.png"
        
        st.image(image=filename_CM,
                caption=f'Confusion Matrix des ausgewählten Modells',
                width=700)

        
    image_url = st.text_input('Link zum Bild', 'https://www.datocms-assets.com/128928/1742429537-colin-powell-main.jpg?auto=compress%2Cformat&fit=crop&h=640&w=960')
    try: 
        og, og_marked, crop = face_detection.get_face(image_url)

        if 'PCA + SVM' in selected_model:
            col_SL, col_SR = st.columns([1,1])
            col_SL1, col_SL2, col_SR1, col_SR2 = st.columns([1,1,1,1])

            with col_SL:
                st.info('Face Detection')
                with col_SL1:
                    try:
                        fig = plt.figure()
                        plt.title('Original mit erkanntem Gesicht')
                        plt.xlabel(f'$w$')
                        plt.ylabel(f'$h$')
                        plt.imshow(og_marked)
                        st.pyplot(fig)
                    except:
                        pass
                
                with col_SL2:
                    try:
                        fig = plt.figure()
                        plt.title('Crop des erkanntes Gesichts')
                        plt.xlabel(f'$w$')
                        plt.ylabel(f'$h$')
                        plt.imshow(crop)
                        st.pyplot(fig)
                    except:
                        pass
                
            with col_SR:
                st.info('Face Recognition')
                with col_SR1:
                    with st.spinner('Berechne Ergebnisse...', show_time=True):
                        try:
                            pca, svm, X_2D = face_recognition.apply_model_A(selected_model=selected_model, crop=crop)

                            X_2D_pca_inv = pca.inverse_transform(X_2D)
                            X_4D_pca_inv = X_2D_pca_inv.reshape(100, 75, 3)
                            fig = plt.figure()
                            plt.title('Eigen-Bild')
                            plt.xlabel(f'$w$')
                            plt.ylabel(f'$h$')
                            plt.imshow(X_4D_pca_inv)
                            st.pyplot(fig)
                        except:
                            pass
                        
                    with col_SR2:
                        try:
                            y_id = np.load('dataset/Target_ID.npy')
                            y_names = np.load('dataset/Target_Names.npy')
                            
                            y_pred = svm.predict(X_2D)
                            
                            mask = np.isin(np.unique(y_id), y_pred)
                            
                            st.write(f'Prediction:')       
                            st.write(y_names[mask])
                        except:
                            pass
        
        elif 'PCA + CS' in selected_model:
                        
            col_SL, col_SR = st.columns([1,1])
            col_SL1, col_SL2, col_SR1, col_SR2 = st.columns([1,1,1,1])

            with col_SL:
                st.info('Face Detection')
                with col_SL1:
                    try:
                        fig = plt.figure()
                        plt.title('Original mit erkanntem Gesicht')
                        plt.xlabel(f'$w$')
                        plt.ylabel(f'$h$')
                        plt.imshow(og_marked)
                        st.pyplot(fig)
                    except:
                        pass
                
                with col_SL2:
                    try:
                        fig = plt.figure()
                        plt.title('Crop des erkanntes Gesichts')
                        plt.xlabel(f'$w$')
                        plt.ylabel(f'$h$')
                        plt.imshow(crop)
                        st.pyplot(fig)
                    except:
                        pass
                
            with col_SR:
                st.info('Face Recognition')
                with col_SR1:
                    with st.spinner('Berechne Ergebnisse...', show_time=True):
                        try:
                            pca, X_2D, X_train_pca, y_train, y_all, relevant_labels, target_names_all, n_targets = face_recognition.preprocess_cs(selected_model=selected_model, crop=crop)

                            X_2D_pca_inv = pca.inverse_transform(X_2D)
                            X_4D_pca_inv = X_2D_pca_inv.reshape(100, 75, 3)
                            fig = plt.figure()
                            plt.title('Eigen-Bild')
                            plt.xlabel(f'$w$')
                            plt.ylabel(f'$h$')
                            plt.imshow(X_4D_pca_inv)
                            st.pyplot(fig)
                            
                        except:
                            pass
                    
                with col_SR2:
                    try:

                        mask = face_recognition.calculate_cs(X_2D=X_2D,
                                                    X_train_pca=X_train_pca,
                                                    y_train=y_train,
                                                    y_all=y_all,
                                                    relevant_labels=relevant_labels,
                                                    n_targets=n_targets)

                        st.write(f'Prediction:')       
                        st.write(target_names_all[mask])
                    except:
                        pass
            
        else:
            st.write("Kein Modell ausgewählt.") 
            
    except:
        st.info('Kein Gesicht erkannt.')