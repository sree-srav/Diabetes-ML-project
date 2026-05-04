# Predicting 30-Day Readmission Risk in Diabetic Patients

A full end-to-end machine learning project on a 100,000+ record clinical dataset to predict 30-day hospital readmission risk in diabetic patients. The project covers data cleaning, exploratory data analysis, hypothesis testing, feature engineering, multi-model training, and performance comparison.

---

## Problem Statement

Diabetic patients have 30-day readmission rates ranging from 14.4% to 22.7% — significantly higher than the general hospitalized population. Early identification of high-risk patients enables targeted interventions that can improve outcomes and reduce healthcare costs.

**Research Question:** What factors significantly contribute to the readmission of diabetic patients, and how can predictive analytics be used to identify and mitigate these risks?

---

## Dataset

- **Source:** [Kaggle — Diabetic Patients Readmission Prediction](https://www.kaggle.com/datasets/saurabhtayal/diabetic-patients-readmission-prediction/data?select=diabetic_data.csv)
- **Size:** 100,000+ patient encounters, 50 features
- **Target variable:** `readmitted` — encoded as binary: `<30` days = 1 (high risk), `NO` or `>30` days = 0
- **Key features:** age, gender, race, time in hospital, number of lab procedures, number of medications, number of diagnoses, discharge disposition, admission type, insulin/medication changes

---

## Project Workflow

### 1. Data Cleaning
- Replaced `?` placeholders with `NaN`
- Dropped high-missing columns (`weight`, `payer_code`, `medical_specialty`)
- Removed rows with missing values in critical fields (`race`, `diag_1`, `diag_2`, `diag_3`)
- Converted age from interval strings (e.g., `[60-70)`) to integers

### 2. Exploratory Data Analysis
- Age distribution — highest patient concentration in the 60–80 age range
- Readmission rates by age group — highest readmission in patients in their 70s
- Readmission rates by race — Caucasian patients had the highest absolute counts
- Readmission rates by gender — females had higher counts; attributed to higher female representation in the dataset
- Distribution of time in hospital — most patients stayed 2–4 days
- Correlation heatmap of all numerical features

### 3. Hypothesis Testing
| Hypothesis | Test Used | Result |
|---|---|---|
| Mean hospital stay differs by gender | Z-test + T-test | Rejected H₀ (p << 0.05) |
| Mean hospital stay differs across age groups | One-way ANOVA | Rejected H₀ (p << 0.05) |
| Weight category is associated with readmission status | Chi-square contingency | Rejected H₀ (p = 0.037) |

### 4. Feature Engineering
- Applied `LabelEncoder` to all categorical columns
- Binary encoded target variable: `<30` → 1, `NO` / `>30` → 0
- Dropped non-predictive identifiers (`encounter_id`, `patient_nbr`)

### 5. Model Training & Evaluation
All models trained on 80/20 stratified train-test split.

| Model | Accuracy |
|---|---|
| **Gradient Boosting Classifier** | **88.86%** |
| Support Vector Machine | 88.84% |
| Random Forest Classifier | 88.78% |
| K-Nearest Neighbors | 87.86% |

Each model was evaluated using accuracy score, classification report (precision, recall, F1), confusion matrix, and precision-recall curve.

---

## Key Findings

- Gradient Boosting was the best-performing model at 88.86% accuracy
- Age group, number of inpatient visits, time in hospital, and number of diagnoses were among the strongest predictors of readmission
- Class imbalance (readmitted <30 days is a minority class) was identified as a factor impacting minority class recall — a known limitation

---

## Tech Stack

`Python` `pandas` `NumPy` `scikit-learn` `matplotlib` `seaborn` `scipy` `statsmodels`

---

## How to Run

1. Clone the repo and install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels
```

2. Download the dataset from Kaggle and place `diabetic_data.csv` in the project root.

3. Run the notebook or script:
```bash
jupyter notebook diabetes_readmission.ipynb
```

---

## Limitations

- Dataset sourced from a single hospital system — generalizability is limited
- Weight column had 97% missing values and was excluded from the main model
- Class imbalance was not addressed with resampling techniques (future improvement)
- Model not validated for clinical deployment

---

## Note

This project was developed for educational and analytical purposes and does not represent a deployed clinical decision support system.
