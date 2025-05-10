# 🧠 Autism Spectrum Disorder (ASD) Prediction using Machine Learning

This project focuses on building a machine learning-based predictive system to identify the likelihood of Autism Spectrum Disorder (ASD) in individuals based on answers to screening questions and demographic data.

Rather than aiming only for high accuracy, the goal of this model is **high recall (sensitivity)** and **balanced F1-score** for **Class 1 (ASD cases)**, which is essential in a healthcare screening context where **missing a positive case is riskier than a false alarm**.

---

## 📌 Problem Statement

Early diagnosis of Autism Spectrum Disorder (ASD) plays a vital role in ensuring timely intervention and support. This project uses machine learning to predict the presence of ASD based on questionnaire responses (AQ-10) and basic personal information such as age, gender, and medical history.

---

## 📊 Dataset Overview

- **Source**: [Kaggle ASD Screening Dataset](https://www.kaggle.com/datasets/fabdelja/autism-screening-adults)
- **Samples**: 800 individuals
- **Features**:
  - **AQ-based scores**: A1 to A10
  - **Demographics**: age, gender, ethnicity, country
  - **Medical history**: jaundice, family history of autism
  - **Screening result** and **test relation**
  - **Target**: `Class/ASD` → 1 = ASD, 0 = Not ASD

---

## ⚙️ Project Workflow

### 1. **Exploratory Data Analysis (EDA)**
- Identified outliers in `age` and `result`
- Found class imbalance: Only **20%** were Class 1 (ASD)
- Used heatmaps and distributions to understand feature behavior
- Strongest predictors: `A6_Score`, `A4_Score`, `result`

### 2. **Preprocessing**
- Replaced missing values (`?`, `others`) with standard categories
- Applied **Label Encoding** to categorical variables and saved encoders using `pickle`
- Removed outliers using IQR-median technique
- Applied **SMOTETomek** to handle class imbalance in training data

### 3. **Model Training**
Trained and compared 3 models:
- Logistic Regression
- Random Forest
- XGBoost

### 4. **Model Tuning**
- Performed **hyperparameter tuning** using `RandomizedSearchCV` on XGBoost
- Tuned for **ROC AUC**, not accuracy, to better reflect real-world screening needs

### 5. **Threshold Tuning**
- Plotted **precision-recall vs threshold**
- Identified optimal threshold using **F1-score maximization**
- Final threshold set to **0.65** instead of default 0.5

---

## 📊 Project Highlights

| Metric                      | Value       | Description |
|-----------------------------|-------------|-------------|
| ✅ **CV ROC AUC (5-fold)**  | **0.96**     | Excellent generalization across folds |
| ✅ **Test ROC AUC**         | **0.86**     | Strong ability to separate ASD vs Non-ASD |
| ✅ **F1 Score (Class 1)**   | **0.61**     | Balanced precision and recall for minority class |
| ✅ **Recall (Class 1)**     | **0.69**     | High sensitivity — catches most ASD cases |
| ✅ **SMOTE + TomekLinks**  | Used         | Smart sampling strategy to handle class imbalance |
| ✅ **Threshold Tuning**     | Applied      | Optimized decision threshold to boost ASD detection |
| ✅ **Hyperparameter Tuning**| Done (XGBoost) | 30-iteration `RandomizedSearchCV` using ROC AUC |
| ✅ **Accuracy**  | 82%       |
> 📌 In health screening, **Test ROC AUC  & Recall (Class 1)** are more valuable than raw accuracy.

---

## 🧪 Classification Report @ Threshold = 0.65

```

Class 0: Precision = 0.92 | Recall = 0.86 | F1-score = 0.89
Class 1 (ASD): Precision = 0.55 | Recall = 0.69 | F1-score = 0.61
Overall Accuracy: 82%
ROC AUC Score: 0.86

````

---

## 🧠 Final Best Model Details

- **Model**: XGBoost Classifier
- **Tuned Hyperparameters**:
```python
{
  'n_estimators': 200,
  'max_depth': 9,
  'learning_rate': 0.1,
  'subsample': 0.8,
  'colsample_bytree': 0.8,
  'scale_pos_weight': 5
}
````

* **Resampling**: SMOTETomek (training set only)
* **Threshold Optimized**: 0.65 (instead of default 0.5)

---

## 💾 Deployment-Ready Assets

* ✅ `best_xgb_model.pkl` — Trained XGBoost model
* ✅ `encoders.pkl` — LabelEncoders for categorical inputs
* ✅ Interactive console app for making predictions
* ✅ Accepts real-time input from user via terminal

---

## 🖥️ How to Run the Predictor

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/asd-prediction-ml.git
   cd asd-prediction-ml
   ```

2. Install required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the prediction system:

   ```bash
   python predict_asd.py
   ```

4. Enter the input values when prompted.

---

## 📌 Key Learnings & Takeaways

* **Threshold tuning** can significantly improve F1 and recall in imbalanced classification.
* **SMOTETomek** is more robust than plain SMOTE — it removes noisy borderline examples.
* **ROC AUC** is a better tuning metric than accuracy for medical classification tasks.
* XGBoost, when properly tuned and threshold-optimized, is extremely effective.

---

## 📈 Future Improvements

* Add SHAP-based feature explanations
* Deploy as a Streamlit or Flask web app
* Train on a larger or more diverse dataset
* Include domain expert feedback in feature selection

---

## 🙌 Acknowledgements

* Dataset: [Kaggle - Autism Screening](https://www.kaggle.com/datasets/fabdelja/autism-screening-adults)
* Libraries: `sklearn`, `xgboost`, `imblearn`, `seaborn`, `matplotlib`, `pandas`

---

## 👨‍💻 Author

**Md Adil Muzaffar**
📧 \[[adilmuzaffar96@gmail.com](mailto:adilmuzaffar96@gmail.com)]
🔗 [www.linkedin.com/in/md-adil-muzaffar](www.linkedin.com/in/md-adil-muzaffar)


---

## 📄 License

This project is licensed under the MIT License. Feel free to use and modify with attribution.

```

---

Let me know if you'd like:

- A matching `requirements.txt` file
- A `predict_asd.py` CLI script template
- A compressed version for resume/portfolio use

I'm happy to help wrap this up in a publish-ready format!
```


Made with 💻 by \[MD ADIL MUZAFFAR]
Feel free to connect or ask questions via \[www.linkedin.com/in/md-adil-muzaffar] or \[adilmuzaffar96@gmail.com].

```

