# 🎯 IT Salary Classifier - Machine Learning Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![ML](https://img.shields.io/badge/ML-Scikit--learn%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-75%25-success.svg)](https://github.com)

A machine learning solution to predict and classify IT job salary levels in Vietnam using NLP, ensemble methods, and feature engineering techniques.

---

## 🏆 Key Results

- **Model Accuracy**: ~75% on test set
- **Best Model**: Voting Ensemble (Random Forest + XGBoost + Gradient Boosting)
- **Dataset**: 1,124 IT job postings from Vietnamese recruitment websites
- **Classification**: 3 salary tiers - Junior (<15M VND) | Middle (15-35M VND) | Senior (>35M VND)

---

## 🎯 Demo: Real-World Predictions

### Case 1: Fresher ReactJS Developer → Junior Tier ✅

![Case 1: Fresher Developer](images/case1.png)

**Input**: "Fresher ReactJS – Mới tốt nghiệp" | Location: Hồ Chí Minh  
**Prediction**: Junior (<15M VND) | **Confidence**: 69.55%  
**Analysis**: Model correctly identified entry-level keywords ("Fresher", "mới tốt nghiệp") and ~1 year experience

---

### Case 2: IT Manager at Large Corp → Senior Tier ✅

![Case 2: IT Manager](images/case2.png)

**Input**: "Trưởng phòng CNTT (IT Manager)" | Company: Tập đoàn lớn | Location: Hà Nội  
**Prediction**: Senior (>35M VND) | **Confidence**: 72.79%  
**Analysis**: Detected management keywords + large company bonus + 8+ years experience inference

---

### Case 3: Java Developer (2 years) → Middle Tier ✅

![Case 3: Java Developer](images/case3.png)

**Input**: "Lập trình viên Java (2 năm kinh nghiệm)" | Location: Đà Nẵng  
**Prediction**: Middle (15-35M VND) | **Confidence**: 62.41%  
**Analysis**: 2-year experience + standard developer role → mid-level classification

---

## 📊 What Drives IT Salaries? Feature Importance

![Feature Importance](images/top20-feature.png)

**Top Salary Predictors:**

1. **Experience & Level** (`exp_years`, `level_score`) - Primary drivers

   - Each additional year → ~1.5-2M VND increase
   - Senior/Manager titles → 2-3x higher salaries

2. **Company Size** (`is_big_company`) - 20-30% premium

   - FPT, Viettel, Banking Groups, Samsung pay significantly more

3. **English Keywords** - Strong indicators for high salaries

   - "Senior", "Manager", "Lead", "Architect" → Senior tier

4. **Location** - Geographic salary variation
   - Hồ Chí Minh: +5-10% | Hà Nội: +3-8% | Other cities: -10-15%

---

## 🔧 Technical Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Data Cleaning                                       │
│ Raw CSV → Salary Parsing (Regex) → KNN Imputation → SQLite  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Feature Engineering                                 │
│ Text Normalization → TF-IDF → Category Extraction → Scaling │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Model Training                                      │
│ SMOTE Balancing → Ensemble Training → Evaluation            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Key Features Engineered

### 1. **Text Processing & NLP**

- **Vietnamese Text Normalization**: Diacritic removal, number stripping
- **TF-IDF Vectorization**: 2500 → 600 features (Chi-Square selection)
- **N-grams**: Captures "machine learning", "data engineer" phrases

### 2. **Structured Features**

| Feature          | Description                    | Example Values                                    |
| ---------------- | ------------------------------ | ------------------------------------------------- |
| `exp_years`      | Experience extracted via regex | 0-15 years                                        |
| `level_score`    | Job seniority (0-5 scale)      | 0=Intern, 1=Junior, 2=Middle, 4=Senior, 5=Manager |
| `is_big_company` | Major corporation flag         | FPT, Viettel, Banks → 1                           |
| `is_english`     | English vs Vietnamese title    | "Senior Dev" → 1, "Lập trình viên" → 0            |
| `job_category`   | Domain classification          | Management, Data/AI, Cloud, QA, Mobile, Dev       |
| `location`       | City (one-hot encoded)         | HCM, Hanoi, Da Nang, etc.                         |

### 3. **Salary Parsing** (Advanced Regex)

Handles diverse formats:

- "10 - 20 Tr VND" → (10, 20)
- "Up to 1500 USD" → (37.5, 37.5) in VND
- "Thỏa thuận" → NaN → KNN imputation

**Total Features**: 654 (600 TF-IDF + 50 categorical + 4 numeric)

---

## 🤖 Models & Techniques

### Ensemble Strategy: Voting Classifier

| Model                 | Hyperparameters              | Role                                   |
| --------------------- | ---------------------------- | -------------------------------------- |
| **Random Forest**     | 200 trees, max_depth=15      | Reduces variance (bagging)             |
| **XGBoost**           | 200 rounds, lr=0.05, depth=6 | Sequential error correction (boosting) |
| **Gradient Boosting** | 100 estimators, lr=0.1       | Conservative learning                  |

**Voting**: Soft voting - averages predicted probabilities from all 3 models

### Handling Imbalanced Data: SMOTE

- **Problem**: Junior (20%) | Middle (65%) | Senior (15%)
- **Solution**: Synthetic Minority Over-sampling
  - Generates synthetic samples between minority class neighbors
  - Balances training set to prevent bias toward Middle class

### Performance Metrics

| Model                  | Accuracy   | F1-Score      | Best Class      |
| ---------------------- | ---------- | ------------- | --------------- |
| Random Forest          | 73-75%     | 0.72-0.74     | Middle          |
| XGBoost                | 74-76%     | 0.73-0.75     | Middle          |
| Gradient Boosting      | 72-74%     | 0.71-0.73     | Middle          |
| **Voting Ensemble** ⭐ | **75-77%** | **0.74-0.76** | **All classes** |

**Per-Class Performance:**

- **Junior** (<15M): Precision 62%, Recall 65% (challenging due to limited samples)
- **Middle** (15-35M): Precision 88%, Recall 85% ⭐ (dominant class, well-balanced)
- **Senior** (>35M): Precision 71%, Recall 68% (good considering class imbalance)

---

## 📁 Project Structure

```
IT_Salary_Classifier/
├── data/
│   └── jobs_it.csv                    # Raw dataset (1,124 records)
├── images/
│   ├── case1.png                      # Demo: Fresher prediction
│   ├── case2.png                      # Demo: Manager prediction
│   ├── case3.png                      # Demo: Developer prediction
│   └── top20-feature.png              # Feature importance chart
├── models/
│   └── wrong_prediction_cases.csv     # Error analysis
├── notebooks/
│   ├── 00_careerviet_data_crawl.ipynb    # Web scraping
│   ├── 01_data_cleaning.ipynb            # Regex parsing, KNN imputation
│   ├── 02_feature_engineering.ipynb      # TF-IDF, feature extraction
│   └── 03_model_training_evaluation.ipynb # SMOTE, ensemble training
└── README.md
```

---

## 🚀 Quick Start

### Installation

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
```

### Run Pipeline

```bash
# Step 1: Data Cleaning
jupyter notebook notebooks/01_data_cleaning.ipynb

# Step 2: Feature Engineering
jupyter notebook notebooks/02_feature_engineering.ipynb

# Step 3: Model Training
jupyter notebook notebooks/03_model_training_evaluation.ipynb
```

---

## 🎓 Key ML Concepts Demonstrated

### Data Engineering

- **Regex Parsing**: Extract structured data from messy text (salary ranges, currencies)
- **KNN Imputation**: Fill missing salaries using similar job characteristics
- **Outlier Detection**: IQR method to remove anomalous salary values

### Feature Engineering

- **NLP for Vietnamese**: Diacritic removal, custom stopwords (ha, noi, hcm)
- **TF-IDF**: Convert job titles to numerical features (2500 terms)
- **Chi-Square Selection**: Reduce to 600 most predictive terms
- **Domain Features**: Extract experience years, company size, job level from text

### Machine Learning

- **SMOTE**: Synthetic oversampling to balance 3-class distribution
- **Ensemble Methods**: Combine bagging (RF) + boosting (XGB, GB) strengths
- **Soft Voting**: Probability averaging for robust predictions
- **Feature Importance**: Interpret model decisions via tree-based rankings

---

## 🔍 Market Insights

**Salary Determinants (ranked by feature importance):**

1. ⭐ **Experience Years** (`exp_years`) - Strongest single predictor
2. 📊 **Career Level** (`level_score`) - Junior vs Senior differentiation
3. 🏢 **Company Prestige** (`is_big_company`) - 20-30% salary premium
4. 🌍 **Location** (HCM/Hanoi) - Urban centers pay 5-25% more
5. 💼 **English Title** (`is_english`) - International positioning indicator
6. 🛠️ **Tech Domain** - Data/AI > Cloud > Development > QA

**Vietnamese IT Salary Reality:**

- **Junior tier** (<15M VND): Entry-level, internships, 0-1.5 years
- **Middle tier** (15-35M VND): 65% of market, 2-5 years experience
- **Senior tier** (>35M VND): Management, 5+ years, specialized skills

**Geographic Variation:**

- Hồ Chí Minh: Highest salaries (tech hub)
- Hà Nội: 2nd highest (government/corporate center)
- Đà Nẵng/Others: 10-15% lower (smaller markets)

---

## 🎯 Why This Project Stands Out

### For Job Interviews:

✅ **Complete ML Pipeline**: Web scraping → Feature engineering → Production model  
✅ **Real-World Complexity**: Messy data (mixed currencies, Vietnamese text, missing values)  
✅ **Advanced Techniques**: SMOTE, ensemble voting, NLP for non-English text  
✅ **Business Value**: Interpretable insights for hiring/salary decisions  
✅ **Technical Depth**: 654 engineered features, 75% accuracy on imbalanced data

### Technical Highlights:

- Custom regex parser for 5+ salary formats
- Vietnamese NLP preprocessing pipeline
- KNN imputation preserving data relationships
- Chi-Square feature selection (76% reduction)
- Soft voting ensemble (3 models)

### Talking Points:

**"How did you handle imbalanced classes?"**  
→ SMOTE synthetic generation + soft voting ensemble

**"What was the hardest challenge?"**  
→ Parsing Vietnamese salary strings with mixed currencies and text states ("Thỏa thuận")

**"How would you improve accuracy?"**  
→ More Senior/Junior samples, BERT for Vietnamese text, company metadata enrichment

---

## 📊 Sample Data

```csv
Job Title,Company,Salary,Location
Trưởng nhóm Lập trình Java,Tổng Công ty Cổ phần Công trình Viettel,30 Tr - 40 Tr VND,Hà Nội
Network Engineer,KDDI Vietnam,11 Tr - 13 Tr VND,Hồ Chí Minh
Senior Data Analyst,FPT Software,25 Tr - 35 Tr VND,Đà Nẵng
```

---

## 📄 License

Educational project for portfolio and academic purposes.

---

**Project Summary**: 75% accuracy | 654 features | SMOTE + Ensemble | Vietnamese NLP | 1,124 job postings

> **Note**: Full technical documentation available in `README_FULL.md` (detailed formulas, theoretical foundations, installation guides).
