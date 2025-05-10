# 🧠 Autism Spectrum Disorder (ASD) Prediction using Machine Learning

This project aims to build a reliable and interpretable machine learning system to predict the likelihood of Autism Spectrum Disorder (ASD) using behavioral and demographic screening data. The model focuses on **high sensitivity (recall)** and **balanced performance**, making it well-suited for **screening** or **early-detection tools** in healthcare applications.

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

---

## 📁 Dataset Overview

- Source: Autism Screening Adult Dataset (Kaggle)
- Size: 800 samples
- Features: 
  - 10 AQ-based screening scores (A1 to A10)
  - Demographics: age, gender, ethnicity, country
  - Medical history: jaundice, family ASD history
  - Target: `Class/ASD` (0 = No ASD, 1 = ASD)

---

## ⚙️ Workflow

### 1. **Exploratory Data Analysis (EDA)**
- Identified data imbalance in the target variable (Class 1 = ~20%)
- Found outliers in `age` and `result`, handled via median replacement
- Cleaned and encoded categorical variables (`LabelEncoder`)
- Feature importance visualized via correlation and later XGBoost importances

### 2. **Preprocessing**
- Handled outliers using IQR method
- Addressed missing or inconsistent values
- Used **SMOTE + TomekLinks** to balance the training set (Class 0: 504, Class 1: 504)

### 3. **Model Training & Evaluation**
- Models compared: Logistic Regression, Random Forest, XGBoost
- Final model: ✅ **XGBoost** with tuned hyperparameters
- Best parameters:
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

### 4. **Threshold Optimization**

* Default threshold (0.5) optimized using Precision-Recall curves
* Final tuned threshold: **0.65**, yielding better F1 and recall

---

## 🔍 Final Evaluation

```text
Classification Report @ threshold=0.65

Class 0 - Precision: 0.92, Recall: 0.86, F1-score: 0.89  
Class 1 - Precision: 0.55, Recall: 0.69, F1-score: 0.61  
Overall Accuracy: 82%  
ROC AUC: 0.86
```

---

## 🔐 Deployment Ready

✅ **Model saved** as `best_xgb_model.pkl`
✅ **Encoders saved** as `encoders.pkl`
✅ Includes a **console-based prediction system** that takes feature input and returns prediction with probability.

---

## 🚀 How to Use

1. Clone the repo and install required libraries:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the predictive script:

   ```bash
   python predict_asd.py
   ```

3. Enter feature values when prompted.

---

## 📌 Conclusion

This project goes beyond accuracy to deliver a **real-world, recall-optimized, cross-validated** model for detecting ASD. The careful treatment of class imbalance, feature engineering, and decision threshold tuning ensures that the model is both **practical** and **reliable** for healthcare screening purposes.

---

## 📧 Contact

Made with 💻 by \[Your Name]
Feel free to connect or ask questions via \[LinkedIn] or \[Email].

```

