# ml-assignment-2
ML assignment 2 classification model + streamlit app 

## a. Problem Statement

The objective of this project is to build and evaluate machine learning models for a binary classification problem.  
Given a set of numerical features extracted from URL-based data, the task is to accurately predict whether a given URL is benign or malicious.

The problem is formulated as a supervised learning task. Multiple classification algorithms are trained and compared using standard evaluation metrics such as Accuracy, Precision, Recall, F1-score, Area Under the ROC Curve (AUC), and Matthews Correlation Coefficient (MCC).

---

## b. Approach Overview

The solution follows a structured machine learning workflow:
- Exploratory Data Analysis (EDA) to understand data characteristics and feature behavior
- Identification of text-based and numerical features
- Data preprocessing, including feature selection and scaling where required
- Training multiple machine learning classification models
- Evaluating and comparing model performance using consistent metrics
- Selecting the best-performing model based on empirical results

---

## c. Dataset Description [1 Mark]

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

- **Text / Identifier Columns (5):**  
  `FILENAME`, `URL`, `Domain`, `TLD`, `Title`  
  These columns are excluded from model training.

- **Numerical Columns (51):**  
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