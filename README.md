---

# 📘 Salary Prediction Model for Data & Analytics Roles

Predicting yearly salary from job title, job platform, and location using ML pipelines with cross-validated hyperparameter tuning.

---

## 📖 Overview

This project builds a machine learning system that predicts **yearly salary** for Data & Analytics job postings.
It is designed as a **real-world portfolio project**, including:

* Full data cleaning
* Feature engineering
* Exploratory Data Analysis (EDA)
* ML pipelines
* Random Forest + Linear Regression comparison
* Hyperparameter tuning with RandomizedSearchCV
* Cross-validation
* Final deployed prediction function

The entire workflow follows best practices used in professional data science teams.

---

## 🧱 Project Structure

```
├── data/
│   └── gsearch_jobs.csv (Kaggle datasource.txt)
├── notebooks/
│   └── salary_prediction.ipynb
├── models/
│   └── tuned_rf_pipeline.pkl
└── README.md
```

---

## 🧹 Data Cleaning & Preprocessing

Key preprocessing steps:

### **1. Title normalization**

Examples:

| Raw Title                 | Normalized Title |
| ------------------------- | ---------------- |
| Data Analyst II           | Data Analyst     |
| Senior Data Analyst       | SR Data Analyst  |
| Data Scientist (Contract) | Data Scientist   |

### **2. Location standardization**

* Trim/clean spacing
* Extract state (e.g., “NY”)
* Identify remote roles
* Extract country

### **3. Clean job platform**

`"via LinkedIn"` → `"LinkedIn"`

### **4. Outlier removal**

Salaries <1st percentile and >99th percentile removed to stabilize training.

---

## 🔍 Exploratory Data Analysis (EDA)

Key visualizations include:

* Salary distribution histograms
* Top job titles (with salary aggregation)
* Top locations
* Platform-level salary trends
* Correlation heatmap
* Distribution plots
* Feature importance

These were created using Matplotlib and Seaborn.

---

## 🧠 Machine Learning Approach

This project evaluates two baseline models:

* **Linear Regression** (baseline)
* **Random Forest Regression** (nonlinear benchmark)

After initial evaluation, the workflow implements:

### **✔ Full ML Pipeline:**

* Imputation (SimpleImputer)
* Ordinal encoding for categorical variables
* Random Forest modeling

All preprocessing happens inside the pipeline (no leakage, no NaN issues).

---

## ⚙️ Hyperparameter Tuning with RandomizedSearchCV

To maximize performance, the model uses:

* **5-fold cross-validation**
* **RandomizedSearchCV** over a rich search space:

  * `n_estimators`
  * `max_depth`
  * `min_samples_split`
  * `min_samples_leaf`
  * `max_features`
  * `bootstrap`

This approach improves performance significantly compared to the default Random Forest.

### Results:

| Model             | RMSE       | MAE        | R²            |
| ----------------- | ---------- | ---------- | ------------- |
| Linear Regression | High       | High       | ~0.04         |
| Default RF        | 28k        | 16k        | ~0.38         |
| **Tuned RF**      | **23–25k** | **14–15k** | **0.48–0.60** |

---

## 📈 Visualizations of Model Tuning

This project includes:

### **1. Distribution of RMSE Across Search Iterations**

Shows model stability and performance improvements.

### **2. RMSE by Iteration Plot**

Illustrates how tuning converges toward better models.

### **3. Feature Importance Plot**

Indicates what drives salary the most (title group, location, platform).

---

## 🛠️ Final Model Pipeline

A complete end-to-end pipeline is used for the final model:

```
ColumnTransformer(
    SimpleImputer → OrdinalEncoder
)
       ↓
RandomizedSearchCV(RandomForestRegressor)
       ↓
Predict Salary
```

The preprocessing + model tuning are combined into a single reproducible workflow.

---

## ▶️ Prediction Function (Ready for Deployment)

```python
def predict_salary(title, platform, location):
    sample = pd.DataFrame({
        "title": [title],
        "job_platform": [platform],
        "location": [location]
    })
    return random_search.best_estimator_.predict(sample)[0]
```

Example:

```python
predict_salary("Data Analyst", "LinkedIn", "United States")
```

---

## 🧪 Cross-Validation Results

The project implements 5-fold CV:

```python
KFold(n_splits=5, shuffle=True, random_state=42)
```

Metrics captured:

* RMSE
* MAE
* R²

This ensures rigorous, industry-grade model validation.

---

## 🚀 Future Enhancements

* Add NLP features using TF-IDF or BERT embeddings
* Deploy the model via:

  * Streamlit App
  * FastAPI / Flask API
  * Hugging Face Space
* Try new models for even better accuracy
* Incorporate skill extraction from job descriptions

---

## 🛡️ License

Released under the **MIT License**.

---
### Saving and Loading the Model

The final tuned Random Forest pipeline is saved using joblib:

joblib.dump(random_search.best_estimator_, "models/tuned_rf_pipeline.pkl")

To load and use the model:

model = joblib.load("models/tuned_rf_pipeline.pkl")
prediction = model.predict(sample)[0]
