# Vaccination-Behavior-ML-Project
# Overview

This project builds machine-learning models to predict whether an individual received the **seasonal flu vaccine**, using survey-based data containing demographics, health information, and behavioral indicators.  
The goal is to understand what factors influence vaccination behavior and how well ML models can perform this classification task.

---

# Dataset

- **~26,700 participants**
- **36 features**
  - 12 categorical  
  - 24 numerical  
- **Target Variable:**
  - **Seasonal Flu Vaccinated** (46.56%)  
  - **Seasonal Flu Unvaccinated** (53.44%)
- **Secondary Variable:**  
  - H1N1 vaccination status

### Potential Use Cases
- Identify groups unlikely to vaccinate  
- Improve targeted public-health messaging  
- Understand behavioral and demographic predictors of vaccine uptake  

---

# Feature Categories

### Demographics
- Age group  
- Sex  
- Race / ethnicity  
- Education level  
- Income-to-poverty ratio  
- Marital status  

### Health & Access Indicators
- Chronic medical condition  
- Healthcare worker  
- Has health insurance  
- Doctor recommendation for flu vaccine  

### Behavioral Indicators
- Wore a face mask  
- Avoided crowds and gatherings  
- Washed hands frequently  
- Stayed home / avoided contact  

---

# Preprocessing & Feature Selection

- Removed columns with **>10,000 missing values**
- Removed rows with remaining NA values
- Feature selection methods used:
  - **SelectKBest** → selected **26–27 features**
  - **PCA** → dimensionality reduction  
- Encoding methods tested:
  - **OrdinalEncoder**
  - **OneHotEncoder**

---

# Models Evaluated

- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest Classifier  
- Gradient Boosting Classifier  
- **Histogram Gradient Boosting Classifier (HGBC)**  
- Multilayer Perceptron (MLP)

---

# Best Performing Model: HGBC

### Initial Cross-Validation
- **Training Accuracy:** 82.12%  
- **Testing Accuracy:** 79.41%

### Hyperparameter Optimization
- **HalvingRandomizedSearchCV**  
- **RandomizedSearchCV** (1000 iterations)

**Final Performance (27 features):**
- **Training Accuracy:** 79.93%  
- **Testing Accuracy:** 78.70%

---

# Additional Analysis

### Class-Weight Adjustment
To reduce **false positives**, HGBC class weights were modified.

- Testing accuracy dropped to **74.55%**,  
  but resulted in a preferred error pattern depending on application needs.

### Pipelines Compared
- OrdinalEncoder + SelectKBest + HGBC  
- OrdinalEncoder + PCA + HGBC  
- OneHotEncoder + SelectKBest + HGBC  
- OneHotEncoder + PCA + HGBC  
- OneHotEncoder + PCA + LR / SVM  

HGBC consistently produced the strongest performance across pipelines.

---

# Summary

- **HGBC** was the top-performing classifier.  
- Behavioral, demographic, and health-access features contribute meaningfully to predictions.  
- **SelectKBest** feature selection improved model performance.  
- Adjusting class weights allows tuning depending on whether false positives or false negatives are more important.
