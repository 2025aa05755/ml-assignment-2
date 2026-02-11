# ml-assignment-2
ML assignment 2 classification model + streamlit app 

## a. Problem Statement

The objective of this project is to build and evaluate machine learning models for a binary classification problem.  
Given a set of numerical features extracted from URL-based data, the task is to accurately predict whether a given URL is benign or malicious.

The problem is formulated as a supervised learning task. Multiple classification algorithms are trained and compared using standard evaluation metrics such as Accuracy, Precision, Recall, F1-score, Area Under the ROC Curve (AUC), and Matthews Correlation Coefficient (MCC).

---

##  Approach Overview

The solution follows a structured machine learning workflow:
- Exploratory Data Analysis (EDA) to understand data characteristics and feature behavior
- Identification of text-based and numerical features
- Data preprocessing, including feature selection and scaling where required
- Training multiple machine learning classification models
- Evaluating and comparing model performance using consistent metrics
- Selecting the best-performing model based on empirical results

---

## b. Dataset Description 

### Dataset Source
The dataset is obtained from the UCI Machine Learning Repository:

- **Dataset Name:** PhiUSIIL Phishing URL Dataset  
- **Source URL:** https://archive.ics.uci.edu  
- **Total Instances:** 235,795 URLs  
- **Total Features:** 56  

### Dataset Structure

- The dataset consists of labeled instances for a binary classification task.
- Each instance represents a URL along with extracted structural, lexical, and content-based features.
- The target variable `label` indicates the class of the URL.
- No missing values are present in the dataset.

### Target Variable Distribution

| Class Label | Count  | Percentage (%) |
|------------|--------|----------------|
| 1 (Phishing) | 134,850 | 57.19 |
| 0 (Legitimate) | 100,945 | 42.81 |

The dataset shows moderate class imbalance, which is suitable for applying standard classification metrics.

---

### Feature Categories

- **Text / Identifier Columns (5 columns):**  
  `FILENAME`, `URL`, `Domain`, `TLD`, `Title`  
  These columns are excluded from model training.

- **Numerical Columns (51 features):**  
  Used as input features for machine learning models.

---

### Features and Data Types

| Feature Name | Data Type |
|-------------|----------|
| URLLength | int64 |
| DomainLength | int64 |
| IsDomainIP | int64 |
| URLSimilarityIndex | float64 |
| CharContinuationRate | float64 |
| TLDLegitimateProb | float64 |
| URLCharProb | float64 |
| TLDLength | int64 |
| NoOfSubDomain | int64 |
| HasObfuscation | int64 |
| NoOfObfuscatedChar | int64 |
| ObfuscationRatio | float64 |
| NoOfLettersInURL | int64 |
| LetterRatioInURL | float64 |
| NoOfDegitsInURL | int64 |
| DegitRatioInURL | float64 |
| NoOfEqualsInURL | int64 |
| NoOfQMarkInURL | int64 |
| NoOfAmpersandInURL | int64 |
| NoOfOtherSpecialCharsInURL | int64 |
| SpacialCharRatioInURL | float64 |
| IsHTTPS | int64 |
| LineOfCode | int64 |
| LargestLineLength | int64 |
| HasTitle | int64 |
| DomainTitleMatchScore | float64 |
| URLTitleMatchScore | float64 |
| HasFavicon | int64 |
| Robots | int64 |
| IsResponsive | int64 |
| NoOfURLRedirect | int64 |
| NoOfSelfRedirect | int64 |
| HasDescription | int64 |
| NoOfPopup | int64 |
| NoOfiFrame | int64 |
| HasExternalFormSubmit | int64 |
| HasSocialNet | int64 |
| HasSubmitButton | int64 |
| HasHiddenFields | int64 |
| HasPasswordField | int64 |
| Bank | int64 |
| Pay | int64 |
| Crypto | int64 |
| HasCopyrightInfo | int64 |
| NoOfImage | int64 |
| NoOfCSS | int64 |
| NoOfJS | int64 |
| NoOfSelfRef | int64 |
| NoOfEmptyRef | int64 |
| NoOfExternalRef | int64 |
| label (Target) | int64 |

---

### Dataset Suitability

- The dataset contains a large number of samples and rich numerical features.
- Exploratory Data Analysis indicates the presence of outliers and multicollinearity among some features.
- Both linear and non-linear classification models are suitable for this dataset.
- The dataset supports a comprehensive comparative study of multiple machine learning algorithms.

## c. Models used

| ML Model Name              | Accuracy  | AUC      | Precision | Recall   | F1       | MCC      |
|----------------------------|-----------|----------|-----------|----------|----------|----------|
| Logistic Regression        | 0.999859  | 1.000000 | 0.999753  | 1.000000 | 0.999876 | 0.999711 |
| Decision Tree              | 0.999943  | 1.000000 | 1.000000  | 0.999901 | 0.999951 | 0.999885 |
| kNN (k=5)                  | 0.998360  | 0.999613 | 0.997730  | 0.999407 | 0.998568 | 0.996652 |
| Gaussian Naive Bayes       | 0.999576  | 0.999835 | 0.999753  | 0.999506 | 0.999629 | 0.999134 |
| Random Forest (Ensemble)   | 0.999943  | 1.000000 | 0.999901  | 1.000000 | 0.999951 | 0.999885 |
| XGBoost (Ensemble)         | 1.000000  | 1.000000 | 1.000000  | 1.000000 | 1.000000 | 1.000000 |

## Observations on Model Performance

| ML Model Name              | Observation about model performance |
|----------------------------|---------------------------------------|
| Logistic Regression        | Achieved extremely high Accuracy (0.999859) and perfect AUC (1.0000), indicating strong linear separability of the dataset. Precision (0.999753) and Recall (1.0000) show balanced classification performance with almost no false negatives. High MCC (0.999711) confirms robust performance even under moderate class imbalance. |
| Decision Tree              | Achieved near-perfect metrics with Accuracy (0.999943) and AUC (1.0000). Perfect Precision (1.0000) suggests zero false positives on the test set. Slightly lower Recall (0.999901) indicates minimal misclassification. While single trees can overfit, the minimal train–test gap suggests stable generalization for this dataset. |
| kNN (k=5)                  | Produced slightly lower Accuracy (0.998360) and MCC (0.996652) compared to other models. AUC (0.999613) remains very high, indicating strong discriminative ability. Lower Precision (0.997730) suggests relatively more false positives than ensemble methods. Performance reflects sensitivity to feature scaling and local neighborhood structure. |
| Gaussian Naive Bayes       | Achieved high Accuracy (0.999576) and AUC (0.999835), demonstrating strong probabilistic separation. Despite the conditional independence assumption, Precision (0.999753) and Recall (0.999506) remain balanced. Slightly lower MCC (0.999134) compared to ensemble methods indicates marginally reduced robustness. |
| Random Forest (Ensemble)   | Delivered very high Accuracy (0.999943) and perfect AUC (1.0000). Balanced Precision (0.999901) and Recall (1.0000) indicate excellent class discrimination. High MCC (0.999885) confirms strong performance under class imbalance. Ensemble averaging reduces variance and improves stability compared to a single Decision Tree. |
| XGBoost (Ensemble)         | Achieved perfect scores across all metrics (Accuracy, AUC, Precision, Recall, F1, MCC = 1.0000). This indicates optimal separation of classes. Regularization (subsampling, L1/L2) and early stopping were applied to prevent overfitting. The gradient boosting mechanism effectively captured complex nonlinear interactions in the dataset. |


### Overall Analysis

All models achieved exceptionally high performance, suggesting that the engineered URL-based features provide strong discriminatory power. Ensemble methods (Random Forest and XGBoost) slightly outperform other classifiers in MCC and AUC, indicating better robustness and generalization. kNN shows comparatively lower MCC due to higher sensitivity to local noise. The near-perfect AUC values across models indicate strong class separability in the feature space.