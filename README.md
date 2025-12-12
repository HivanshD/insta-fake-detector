# Instagram Fake Profile Detector

## Overview

This project builds an **end-to-end machine learning pipeline to detect fake Instagram accounts** using account metadata. The task is framed as a **binary classification problem** (fake vs. real), with a focus on **reliable detection of fake accounts** while maintaining strong overall performance.

The project demonstrates a clean ML workflow including exploratory analysis, feature engineering, model comparison, threshold optimization, and final evaluation on a held-out test set.

---

## What This Project Shows

- Clear problem formulation and data splitting
- Thoughtful feature engineering driven by data analysis
- Fair model comparison using consistent preprocessing
- Explicit threshold tuning based on precision–recall tradeoffs
- Clean, reproducible scikit-learn pipelines
- Emphasis on interpretability and evaluation, not just accuracy

---

## Models Evaluated

Three models were trained and validated under identical conditions:

- **Logistic Regression** — interpretable baseline
- **SVM (RBF kernel)** — non-linear decision boundary
- **Random Forest** — captures feature interactions and non-linear effects

The final model was selected based on validation performance and suitability for detecting fake accounts.

---

## Results

Validation performance (representative):

| Model               | Accuracy | Precision | Recall | F1   | ROC-AUC |
|--------------------|----------|-----------|--------|------|---------|
| Logistic Regression | ~0.90    | ~0.96     | ~0.84  | ~0.90 | ~0.97   |
| SVM (RBF)           | ~0.90    | ~0.96     | ~0.84  | ~0.90 | ~0.98   |
| **Random Forest**   | **~0.94**| **~0.96** | **~0.91** | **~0.94** | **~0.98** |

After threshold optimization using the validation precision–recall curve, the final model achieves **high recall for fake accounts** while maintaining strong precision.

Final evaluation is performed on a **held-out test set** to estimate real-world generalization.

---

## Repository Structure
insta-fake-detector/
├── data/
│ └── raw/
│ ├── train.csv
│ └── test.csv
├── notebooks/
│ └── 02_clean_pipeline.ipynb
├── artifacts/ # trained models (optional)
├── src/ # reusable code (optional)
└── README.md


---

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook

---

## Notes

This project prioritizes **clean methodology, interpretability, and evaluation discipline** over unnecessary model complexity. All detailed analysis and visualizations are documented in the notebook
