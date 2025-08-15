# 👤 Face Recognition with PCA, SVM & Cosine Similarity, based on LFW dataset

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
   - Converting to grayscale and normalization  
2. **Model training**  
   - Dimensionality reduction using PCA  
   - Training an SVM classifier  
   - Cosine similarity as an alternative approach without traditional training  
3. **Evaluation**  
   - Accuracy comparison  
   - Confusion matrix visualization  
4. **Deployment**  
   - Streamlit app for interactive testing

---

## 📈 Results
- **SVM model:** approx. X% accuracy  
- **Cosine similarity:** approx. Y% accuracy  
- PCA reduced dimensionality from N to M with minimal accuracy loss.  
- Screenshot of the app:  
  ![Streamlit App Screenshot](results/model_comparison.png)

---

## 🛠️ Installation & Usage
```bash
git clone https://github.com/yourusername/face-recognition-pca-svm.git
cd streamlit-face-recognition-app
pip install -r requirements.txt
streamlit run streamlit_app/main.py
```

---

## 📄 Licenses & Third-Party Dependencies

- This project uses [OpenCV (cv2)](https://opencv.org/), which is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
- The Haar cascade file `haarcascade_frontalface_default.xml` is part of the OpenCV project and is also licensed under the Apache 2.0 License.
