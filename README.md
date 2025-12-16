# Instagram Fake Profile Detector

A machine learning system for detecting fake Instagram accounts using account metadata. Achieves 98.33% recall with 99% ROC-AUC on held-out test data.

**Live Demo:** [Streamlit App](https://hivanshd-insta-fake-detector-app-nnzrl8.streamlit.app/)

**Full Analysis:** [[Open Notebook]](https://nbviewer.org/github/HivanshD/insta-fake-detector/blob/main/notebooks/mainnotebook.ipynb)
---

## Problem Statement & Methodology

### Formulation

Social media platforms face a critical challenge with fake accounts that spread misinformation and engage in fraudulent activities. This project develops a binary classifier to automatically identify fake Instagram profiles using only account metadata.

**Objective:** Maximize recall (catch 98%+ of fakes) while maintaining acceptable precision for human review.

### Approach

**1. Data Understanding**
- Dataset: 694 Instagram profiles from [Kaggle](https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-accounts)
- 347 real accounts, 347 fake accounts
- Split: 574 training (82.7%), 120 test (17.3%)
- 11 raw features: followers, posts, bio length, profile pic, etc.

**2. Feature Engineering**

Created 12 derived features to capture behavioral patterns:

| Category | Features | Rationale |
|----------|----------|-----------|
| Ratios | follower_ratio, following_ratio | Balance between followers/following |
| Engagement | engagement_rate, posts_per_following | Activity level relative to audience |
| Quality | profile_completeness | Completeness score (0-1) |
| Patterns | suspicious_username, very_sparse_profile | Red flags for fake behavior |
| Transforms | log_followers, log_follows, log_posts | Handle skewed distributions |
| Binary | has_posts, has_bio | Simple presence indicators |

Result: 23 total features (engineered features = 57.3% of model importance)

**3. Model Development**

Compared 3 baseline algorithms:

| Model | Accuracy | Recall | F1 | ROC-AUC |
|-------|----------|--------|-----|---------|
| Logistic Regression | 88.70% | 84.21% | 87.57% | 96.25% |
| SVM (RBF) | 83.48% | 75.44% | 80.27% | 92.18% |
| Random Forest | 93.91% | 91.23% | 93.70% | 98.44% |

Selected Random Forest, then optimized with RandomizedSearchCV:
- 30 parameter combinations tested
- 5-fold cross-validation
- Optimization metric: Recall

**Optimal Configuration:**
```
n_estimators: 153
max_depth: None (unlimited)
max_features: log2
min_samples_split: 7
min_samples_leaf: 6
```

**4. Threshold Optimization**

Adjusted decision threshold from 0.5 to 0.4:

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.25 | 85.00% | 100.00% | 92.00% |
| 0.40 | 91.80% | 98.25% | 94.92% |
| 0.50 | 94.74% | 94.74% | 94.74% |

Threshold 0.4 selected to prioritize catching fakes while maintaining reasonable precision.

### Evaluation

**Test Set Performance (120 held-out samples):**

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Accuracy | 91.67% | Correctly classified 110/120 accounts |
| Precision | 86.76% | 87% of flagged accounts are truly fake |
| Recall | 98.33% | Caught 59/60 fake accounts (only 1 missed) |
| F1 Score | 92.19% | Strong balance of precision and recall |
| ROC-AUC | 99.00% | Near-perfect discrimination ability |

**Confusion Matrix:**

|  | Predicted Real | Predicted Fake |
|---|----------------|----------------|
| Actual Real | 51 (TN) | 9 (FP) |
| Actual Fake | 1 (FN) | 59 (TP) |

**Key Results:**
- Only 1 fake account missed out of 60 (1.67% false negative rate)
- 9 false positives (dormant/new accounts) manageable for human review
- System catches 98.33% of all fake accounts

**Cross-Validation Stability:**
- 5-fold CV Accuracy: 92.34% ± 1.93%
- 5-fold CV Recall: 91.97% ± 3.64%
- Train-validation gap: 3.44% (minimal overfitting)

**Error Analysis:**

False Negative (1): Sophisticated fake with 309 followers, 250 following, 34 posts, complete profile. Demonstrates limitation of metadata-only detection.

False Positives (9): Legitimate accounts with unusual patterns (low activity, new users). Easily verified by human review.

---

## Model Architecture

**Pipeline:**
1. Feature Engineering: 11 raw → 23 total features
2. StandardScaler: Normalize numeric features
3. Random Forest: 153 trees, unlimited depth
4. Custom Threshold: 0.40 for predictions

**Top 5 Most Important Features:**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | log_followers | 15.86% |
| 2 | #followers | 15.67% |
| 3 | #posts | 11.06% |
| 4 | log_posts | 10.77% |
| 5 | nums/length username | 7.71% |

---

**requirements.txt:**
```txt
streamlit==1.31.0
pandas==2.1.4
numpy==1.26.3
scikit-learn==1.4.0
joblib==1.3.2
plotly==5.18.0
```

## Results Summary

**Best Performance Metrics:**
- 98.33% Recall (only 1 fake missed)
- 99.00% ROC-AUC (near-perfect discrimination)
- 91.67% Accuracy (110/120 correct)
- 86.76% Precision (manageable false positive rate)

**Model Improvements vs Baseline:**
- Recall: +3.51 percentage points
- Accuracy: +0.87 percentage points
- F1 Score: +1.04 percentage points
- ROC-AUC: +0.68 percentage points


