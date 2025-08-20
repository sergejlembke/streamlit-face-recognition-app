# 👤 Face Recognition with PCA, SVM & Cosine Similarity, based on LFW dataset

Link to app: https://lfw-face-recognition-demo.streamlit.app/

## 🎯 Project Description
This project was developed as the final assignment of my Data Science Bootcamp (2025).  
The goal was to implement, compare, and deploy different approaches for face recognition using an interactive Streamlit app.

---

## 🚀 Features
- **Technologies:** Python, NumPy, scikit-learn, openCV2, Streamlit  
- **Algorithms:** Principal Component Analysis (PCA), Support Vector Machines (SVM), Cosine Similarity  
- **Dataset:** [LFW – Labeled Faces in the Wild](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html) fetched from scikit-learn
- **Outcome:** Interactive face recognition app comparing two model approaches

---

## 📊 Approach
1. **Data preprocessing**  
   - Loading & cleaning the LFW dataset
   - Normalization
2. **Hyperparameter tuning**  
   - Grid search for optimal PCA and SVM parameters
   - Cross-validation for robust evaluation
3. **Model training**  
   - Dimensionality reduction using PCA
   - Training an SVM classifier
   - Cosine similarity as an alternative approach without traditional training
4. **Evaluation**  
   - Accuracy comparison
   - Confusion matrix visualization
5. **Deployment**  
   - Streamlit app for interactive testing

---

## 📈 Results
- **SVM model:** between 74.87% and 95.15% accuracy  
- **Cosine similarity:** between 77.54% and 96.15% accuracy  
- Screenshot of the app:  
  ![Streamlit App Screenshot](app_demo.png)

---

## 🛠️ Installation & Usage
```bash
git clone https://github.com/sergejlembke/streamlit-face-recognition-app.git
cd streamlit-face-recognition-app
pip install -r requirements.txt
cd streamlit_app
streamlit run main.py
```

---

## 📄 Licenses & Third-Party Dependencies

- This project uses [OpenCV (cv2)](https://opencv.org/), which is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
- The Haar cascade file [haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml) is part of the OpenCV project and is also licensed under the Apache 2.0 License.
