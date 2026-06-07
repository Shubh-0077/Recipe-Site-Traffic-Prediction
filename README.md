# 🍽️ Recipe Site Traffic Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge"/>
</p>

> **Predicting which recipes will drive high website traffic for homepage selection — using machine learning to replace guesswork with data-driven decisions.**

---

## 📋 Table of Contents

- [Business Problem](#-business-problem)
- [Project Overview](#-project-overview)
- [Dataset Description](#-dataset-description)
- [Project Workflow](#-project-workflow)
- [Data Validation Summary](#-data-validation-summary)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Machine Learning Models](#-machine-learning-models)
- [Model Performance](#-model-performance)
- [Business Metric](#-business-metric)
- [Key Findings](#-key-findings)
- [Recommendations](#-recommendations)
- [Technologies Used](#️-technologies-used)
- [Installation](#-installation)
- [Repository Structure](#-repository-structure)
- [Author](#-author)

---

## 💼 Business Problem

**Company:** Tasty Bytes — a recipe discovery and subscription platform.

Popular recipes featured on the homepage can increase website traffic by up to **40%**, and higher traffic directly drives subscription growth. However, Tasty Bytes' current homepage recipe selection process is **entirely manual** — with no systematic or data-driven approach to identifying which recipes are likely to perform well.

This creates a high-cost opportunity: if editors choose the wrong recipe, traffic and subscriptions suffer. If they choose the right one, growth accelerates.

**The goal:** Build a machine learning model that predicts whether a recipe will generate **high website traffic** when featured on the homepage, enabling editors to make smarter, data-backed decisions.

**Business Target:** The Product Manager requested a model that correctly identifies high-traffic recipes at least **80% of the time** (i.e., Recall ≥ 80%).

---

## 🔍 Project Overview

This is an end-to-end supervised binary classification project. Starting from raw recipe data, I validated and cleaned the dataset, performed exploratory analysis to understand what drives traffic, built and compared two machine learning models, evaluated them using business-relevant metrics, and delivered actionable recommendations for deployment.

The project follows a structured data science workflow designed to mirror real-world industry practice — from data validation through to business impact framing.

---

## 📦 Dataset Description

| Property | Details |
|---|---|
| **Source** | Recipe Site Traffic Dataset (recipe_site_traffic_2212.csv) |
| **Rows** | 947 recipes |
| **Original Features** | 8 columns |
| **Target Variable** | `high_traffic` (Binary: High / Low) |

### Features

| Feature | Type | Description |
|---|---|---|
| `recipe` | Numeric | Unique recipe identifier |
| `calories` | Numeric | Caloric content of the recipe |
| `carbohydrate` | Numeric | Grams of carbohydrates |
| `sugar` | Numeric | Grams of sugar |
| `protein` | Numeric | Grams of protein |
| `category` | Categorical | Recipe type (10 categories) |
| `servings` | Numeric | Number of servings produced |
| `high_traffic` | Binary Target | Whether the recipe generated high traffic |

**Target Class Distribution:**
- 🟢 High Traffic: 574 recipes (60.6%)
- 🔴 Low Traffic: 373 recipes (39.4%)

---

## 🔄 Project Workflow

```
Raw Data
   │
   ▼
Data Validation & Cleaning
   │  - Missing value treatment
   │  - Duplicate check
   │  - Category validation
   │  - Data type standardization
   │  - Target variable encoding
   ▼
Exploratory Data Analysis
   │  - Traffic distribution
   │  - Category frequency
   │  - Category vs traffic relationship
   │  - Nutritional feature review
   ▼
Feature Engineering & Preprocessing
   │  - Numeric: Median imputation + Standard Scaling
   │  - Categorical: Mode imputation + One-Hot Encoding
   │  - Train/test split (80/20, stratified)
   ▼
Model Development
   │  - Baseline: Logistic Regression
   │  - Comparison: Random Forest (200 estimators)
   ▼
Model Evaluation
   │  - Accuracy, Precision, Recall, F1, ROC-AUC
   │  - Business Metric: High Traffic Capture Rate (Recall)
   ▼
Feature Importance Analysis
   ▼
Business Recommendations
```

---

## ✅ Data Validation Summary

The dataset was systematically validated before any modeling work began.

| Column | Issue Found | Action Taken |
|---|---|---|
| `recipe` | No duplicates | No action needed |
| `calories` | Missing values present | Filled with **median** |
| `carbohydrate` | Missing values present | Filled with **median** |
| `sugar` | Missing values present | Filled with **median** |
| `protein` | Missing values present | Filled with **median** |
| `category` | Checked against 10 expected categories | No invalid categories found |
| `servings` | Mixed data types (strings with text) | Extracted numeric values |
| `high_traffic` | Missing values present | Missing → `"Low"` (no traffic recorded = low traffic) |

**Result:** Post-cleaning dataset contained **zero missing values** across all modeling features.

> <img width="1057" height="523" alt="image" src="https://github.com/user-attachments/assets/57ef170a-93e7-401d-81e1-e8f3bfc1bf43" />


---

## 📊 Exploratory Data Analysis

### Traffic Distribution

The dataset is moderately imbalanced, with 60.6% of recipes generating high traffic. The imbalance is mild enough that no resampling technique (e.g., SMOTE) was required.

> <img width="725" height="577" alt="image" src="https://github.com/user-attachments/assets/7f0cd0e6-9c50-42fd-8b1d-3a8e3e0e9822" />


### Category Distribution

Recipe categories are unevenly distributed. Chicken, Dessert, and One Dish Meal have the highest recipe counts in the dataset.

> <img width="1010" height="643" alt="image" src="https://github.com/user-attachments/assets/c9200c72-a721-48d8-90d3-1ec09036709b" />


### Category vs. Traffic

Traffic performance varies meaningfully across recipe categories. Certain categories — particularly **Chicken** and **Breakfast** — consistently generate high traffic, suggesting recipe type is a strong signal for the model.

> <img width="1007" height="653" alt="image" src="https://github.com/user-attachments/assets/bada2838-71c2-4c0f-8b75-1dffdfada217" />


### Key EDA Observations

- Recipe category is a strong predictor of traffic outcome
- Nutritional features (calories, protein, sugar, carbohydrates) show variation across recipes and likely carry predictive signal
- The class distribution is workable for standard classification without resampling

---

## 🤖 Machine Learning Models

Two models were trained and evaluated using a consistent preprocessing pipeline.

### Preprocessing Pipeline

```python
Numeric Features  →  Median Imputation  →  Standard Scaling
Categorical Features  →  Mode Imputation  →  One-Hot Encoding
```

### Model 1: Logistic Regression (Baseline)

Selected because it is interpretable, fast, and provides a strong linear benchmark for binary classification. Trained with `max_iter=1000` to ensure convergence.

### Model 2: Random Forest (Comparison)

Selected as a non-linear comparison model. Trained with `n_estimators=200` and `random_state=42` for reproducibility. Expected to capture feature interactions that logistic regression cannot.

---

## 📈 Model Performance

| Metric | Logistic Regression | Random Forest |
|---|:---:|:---:|
| **Accuracy** | 77.37% | 73.68% |
| **Precision** | 84.62% | 82.18% |
| **Recall** ⭐ | **76.52%** | 72.17% |
| **F1 Score** | 80.37% | 76.85% |
| **ROC-AUC** | 86.61% | 83.21% |

> ⭐ Recall is the primary business metric — maximising it reduces missed high-traffic recipe opportunities.

**Selected Model: ✅ Logistic Regression**

Logistic Regression outperformed Random Forest across all five evaluation metrics. Notably, it achieved a higher recall (76.52% vs. 72.17%), which directly aligns with the business objective of minimising missed high-traffic recipes.

> <img width="1026" height="253" alt="image" src="https://github.com/user-attachments/assets/64ccdab9-7191-438e-858c-c31b98425840" />


### Feature Importance (from Random Forest)

The top predictive features identified were:
1. **Calories** 
2. **Protein** — highest importance
3. **Sugar**
4. **Carbohydrates**
5. **Recipe Category** (across multiple one-hot encoded columns)

> <img width="1013" height="568" alt="image" src="https://github.com/user-attachments/assets/799d5ea5-afbc-4965-93cb-241385c61316" />


---

## 📐 Business Metric

### High Traffic Capture Rate (Recall)

The primary business objective is to **avoid missing** recipes that would have generated high traffic. A false negative — failing to feature a recipe that would have gone viral — represents a direct lost opportunity for Tasty Bytes in traffic and subscriptions.

Therefore, **Recall** was selected as the primary business metric:

```
High Traffic Capture Rate = True Positives / (True Positives + False Negatives)
```

### Result

| Business Metric | Target | Achieved |
|---|:---:|:---:|
| High Traffic Capture Rate | ≥ 80% | **76.52%** |

The model correctly identifies approximately **77 out of every 100** high-traffic recipes before homepage publication. While this falls slightly short of the 80% target, it represents a significant and measurable improvement over the current manual selection process, which has no systematic basis for predictions.

---

## 💡 Key Findings

1. **Recipe category is a strong predictor.** Traffic performance varies significantly across categories. Chicken and Breakfast recipes consistently generate high traffic, while other categories underperform. This insight alone can guide editorial strategy.

2. **Nutritional features carry predictive signal.** Calories, protein, sugar, and carbohydrates are among the most important features in the Random Forest model — suggesting that recipe composition influences user engagement.

3. **Logistic Regression outperformed Random Forest.** The simpler, linear model generalised better on this dataset, likely because the relationships between features and traffic are closer to linear than non-linear.

4. **The dataset is reasonably balanced.** With a 60/40 class split, standard classification was applied without resampling.

5. **Model achieves 86.61% ROC-AUC**, indicating strong overall discriminative ability — the model ranks high-traffic recipes significantly better than random chance.

---

## 🚀 Recommendations

1. **Deploy Logistic Regression as a decision-support tool.** Use model predictions to rank and shortlist recipe candidates for homepage placement — particularly for editors who currently rely entirely on intuition.

2. **Track Recall as the ongoing business KPI.** Set up a monitoring dashboard that tracks the High Traffic Capture Rate over time as the model is used in production.

3. **Run an A/B test.** Compare model-assisted recipe selection against the current manual process across a defined period. Measure impact on page views, session duration, and subscription conversion rate.

4. **Collect richer engagement data.** User ratings, click-through rates, social shares, time-on-page, and seasonal trends would meaningfully improve future model iterations.

5. **Iterate toward the 80% target.** Hyperparameter tuning, cross-validation, and experimenting with Gradient Boosting (XGBoost / LightGBM) could push recall past the business threshold.

6. **Prioritise high-value categories editorially.** Even without the model, the EDA finding that Chicken and Breakfast recipes consistently outperform others is actionable intelligence for the content team today.

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Data Manipulation** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Models** | Logistic Regression, Random Forest |
| **Preprocessing** | ColumnTransformer, Pipeline, StandardScaler, OneHotEncoder |
| **Notebook** | Jupyter Notebook |
| **Presentation** | PowerPoint (python-pptx) |
| **Version Control** | Git, GitHub |

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/recipe-traffic-prediction.git
cd recipe-traffic-prediction
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Notebook

```bash
jupyter notebook notebooks/recipe_traffic_prediction.ipynb
```

---

## 📁 Repository Structure

```
recipe-traffic-prediction/
│
├── data/
│   └── recipe_site_traffic_2212.csv          # Raw dataset
│
├── notebooks/
│   └── recipe_traffic_prediction.ipynb       # Full analysis notebook
│
├── reports/
│   └── project_report.md                     # Written project report
│
├── presentation/
│   └── Recipe_Traffic_Prediction.pptx        # Stakeholder presentation
│
├── images/
│   ├── traffic_distribution.png              # EDA: target class distribution
│   ├── category_distribution.png             # EDA: recipe categories
│   ├── category_vs_traffic.png               # EDA: category × traffic
│   ├── model_comparison.png                  # Model metric comparison chart
│   └── feature_importance.png               # Top 10 features (Random Forest)
│
├── README.md                                 # Project documentation (this file)
├── requirements.txt                          # Python dependencies
└── .gitignore                                # Files excluded from version control
```


## 👤 Author

**Shubham Malkar**

Data Science Enthusiast | Mumbai, India

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yourusername)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-FF5722?style=for-the-badge&logo=google-chrome&logoColor=white)](https://yourportfolio.com)

---

<p align="center">
  <i>If you found this project helpful, consider giving it a ⭐ — it helps others discover it!</i>
</p>
