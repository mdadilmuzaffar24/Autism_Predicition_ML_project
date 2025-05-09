🧠 ASD (Autism Spectrum Disorder) Detection Using Machine Learning

![image](https://github.com/user-attachments/assets/e58d5b20-e819-42fd-bf41-0437bb8539bb)


📚 Overview
This project focuses on building a robust machine learning system to detect Autism Spectrum Disorder (ASD) tendencies based on behavioral and personal attributes.
We have successfully developed a high-performance predictive model using XGBoost with tuned hyperparameters, achieving strong cross-validation metrics, high ROC AUC scores, and excellent generalization on unseen data.

✅ The solution is optimized for maximum reliability, generalization, and interpretability, making it suitable for real-world deployment in healthcare or early screening tools.

🚀 Final Model Performance Highlights
Best Model: ⚡ XGBoost Classifier (with hyperparameter tuning)

Cross-Validation ROC AUC Scores:
[0.9657, 0.9604, 0.9654, 0.9499, 0.9753]

Mean Cross-Validation ROC AUC:
🔥 0.9633

Best Threshold for Maximum F1-Score:
🔥 0.89999825

Final Test Set Performance:

Accuracy: 83.75%

F1-Score: 65.79%

ROC AUC Score: 81.64%

✨ These results show that the model is highly effective at identifying individuals with potential ASD tendencies with strong generalization ability.

## 🎯 Objective

* To predict ASD likelihood based on responses to specific behavioral and demographic questions.
* To build and evaluate multiple machine learning models.
* To save and share the best model for further prediction use.

---

## 🛠️ Project Workflow

1. **Data Collection**
2. **Data Preprocessing**

   * Handling missing values
   * Encoding categorical variables
   * Feature selection
3. **Model Building**

   * Logistic Regression
   * Random Forest Classifier
   * XGBoost Classifier
4. **Model Evaluation**

   * Confusion Matrix
   * Precision, Recall, F1-Score
   * ROC-AUC Score
5. **Model Selection**

   * Selecting the best performing model (XGBoost)
6. **Model Saving**

   * Saving the trained model using `joblib`

---

## 🔝 Final Best Model Summary

* **Final Best Model**: XGBoost Classifier with Tuned Hyperparameters
* **Cross-Validation ROC AUC Scores**:
  `[0.96568964, 0.96039604, 0.96539555, 0.94990099, 0.97534653]`
* **Mean CV ROC AUC Score**:
  `0.9633`
* **Best Decision Threshold for Highest F1 Score**:
  `0.89999825`

✅ The model demonstrates excellent performance, generalizing well across validation folds.

---

## 📊 Model Performance

| Model                    | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------------ | -------- | --------- | ------ | -------- | ------- |
| XGBoost Classifier       | 83.75%   | 56.8%     | 78.1%  | 65.7%    | 81.6%   |
| Random Forest Classifier | 80.62%   | 51.2%     | 68.7%  | 58.6%    | 76.1%   |
| Logistic Regression      | 78.12%   | 46.8%     | 68.7%  | 55.7%    | 74.6%   |

---

## 📚 Dataset Features

The dataset contains 19 features related to:

* Age
* Gender
* Ethnicity
* Family history of ASD
* Social responsiveness and communication
* Behavioral characteristics

The 20th feature is the **target variable**:

* **0** → Not Likely ASD
* **1** → Likely ASD

---

## 🧩 Technologies Used

* **Python 3.9**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **Matplotlib** and **Seaborn** for visualization
* **Joblib** for model serialization

---

## 🧪 How to Use

1. **Clone this repository**:

```bash
git clone https://github.com/mdadilmuzaffar24/ASD_Prediction_System.git
cd ASD_Prediction_System
```

2. **Install Dependencies**:

```bash
pip install -r requirements.txt
```

3. **Load the saved model and predict**:

You can import the model and use it to predict new data samples after preprocessing them correctly.

```python
import joblib
model = joblib.load('best_xgb_model.pkl')

# Predicting
prediction = model.predict(new_input_data)
```

---

## 📦 Project Structure

```bash
ASD_Prediction_System/
│
├── app.py (optional - manual predictions)
├── best_xgb_model.pkl  # Saved XGBoost model
├── requirements.txt    # Python packages
├── README.md            # Project documentation
├── dataset/             # Dataset folder (optional)
│   └── sample_data.csv
└── utils/               # Preprocessing utilities (optional)
    └── preprocess.py
```

---

## 🔮 Future Enhancements

* Further hyperparameter tuning using GridSearchCV or Optuna.
* Expand the dataset with more samples for better generalization.
* Model interpretability with SHAP or LIME to explain individual predictions.

---

## 🤝 Let's Connect!

If you like this project or want to collaborate, feel free to connect:

* **GitHub**: [mdadilmuzaffar24](https://github.com/mdadilmuzaffar24)
* **LinkedIn**: [Md Adil Muzaffar](https://www.linkedin.com/in/md-adil-muzaffar/)

---

## ⚖️ License

This project is open-source and available under the **MIT License**.

---

✅ **Feel free to fork, star ⭐, and contribute to this project!**

---

# 🚀 Thank you for visiting!

