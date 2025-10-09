# Data Science Concepts - Quick Reference

## Overview

This page provides concise explanations of fundamental data science techniques for data preparation, preprocessing, and transformation. These techniques should be applied before building ML models. Proper data preparation is critical for model performance.

**Related:** See [ML Concepts](ml_concepts.md) for machine learning algorithms, training, and evaluation techniques.

---

## 1. Feature Preprocessing

*Transformations applied to features before model training.*

### Feature Scaling (Normalization)

Scales features to a fixed range, typically [0, 1] using: (x - min) / (max - min). Preserves original distribution shape and relationships. Sensitive to outliers as they determine min/max. Required for distance-based algorithms like KNN, SVM, neural networks. Critical preprocessing step before training models with gradient-based optimization algorithms.

### Standardization (Z-score Normalization)

Transforms features to have mean=0 and standard deviation=1 using: (x - mean) / std. Assumes roughly normal distribution; less sensitive to outliers than min-max scaling. Results in unbounded range (typically -3 to +3 for normal data). Preferred for algorithms assuming normally distributed data.

---

## 2. Data Encoding Techniques

*Methods to convert categorical data into numerical format for machine learning models.*

### One Hot Encoding

Converts categorical variables into binary vectors where only one element is 1 (hot) and others are 0. Each category becomes a separate binary column. For example, colors [Red, Blue, Green] → Red: [1,0,0], Blue: [0,1,0], Green: [0,0,1]. Prevents model from assuming ordinal relationships between categories.

### Label Encoding

Converts categorical labels into integers (0, 1, 2, ...) by assigning each unique category a number. Simple and memory-efficient but implies ordinal relationship between categories. Example: [Red, Blue, Green] → [0, 1, 2]. Best for ordinal data or tree-based models that don't assume ordering.

### Ordinal Encoding

Similar to label encoding but explicitly preserves meaningful order in categorical variables. User assigns specific integers reflecting the inherent ranking (e.g., Low=1, Medium=2, High=3). Appropriate for ordinal features like education level, satisfaction ratings, or size categories. Encodes domain knowledge about category ordering.

### Target Encoding (Mean Encoding)

Replaces each category with the mean of the target variable for that category. For example, if "Blue" items have average price $50, all "Blue" becomes 50. Risk of overfitting; use with cross-validation or smoothing techniques. Effective for high-cardinality categorical features.

### Binary Encoding

Converts categories to integers then to binary code, using fewer columns than one-hot encoding. Each category gets a binary representation (e.g., 5 categories need only 3 binary columns instead of 5). Reduces dimensionality while preserving uniqueness. Useful for high-cardinality features.

---

## 3. Missing Data Imputation

*Techniques to handle missing values in datasets.*

### Simple Imputation (Mean/Median/Mode)

Replaces missing values with mean (continuous), median (skewed continuous), or mode (categorical) of the column. Fast and simple but ignores relationships between features and reduces variance. Mean is sensitive to outliers; median is robust. Mode for categorical data.

### Forward Fill / Backward Fill

Time-series imputation that fills missing values with the previous (forward) or next (backward) observed value. Assumes temporal continuity and that values don't change drastically between time points. Forward fill: copies last known value; backward fill: copies next known value. Common in stock prices, sensor data.

### KNN Imputation

Fills missing values using K-Nearest Neighbors; finds k most similar samples and uses their average. Considers feature relationships unlike simple imputation, but computationally expensive for large datasets. Distance metric (usually Euclidean) determines similarity. Typical k values: 3-10.

### Multiple Imputation

Statistical technique that creates multiple plausible imputed datasets, analyzes each, then pools results. Accounts for uncertainty in missing values unlike single imputation methods. MICE (Multivariate Imputation by Chained Equations) is popular implementation. Produces more accurate standard errors and confidence intervals.

### Imputation with Supervised Learning

Treats missing value prediction as a machine learning problem using other features as predictors. Train regression model (for continuous) or classification model (for categorical) on complete cases. More sophisticated than mean/median but risk of overfitting. Can capture complex feature relationships.

---

## 4. Sampling Techniques for Imbalanced Data

*Methods to address class imbalance in classification problems.*

### SMOTE (Synthetic Minority Over-sampling)

Creates synthetic samples for minority class by interpolating between existing minority samples and their nearest neighbors. Generates new points along line segments connecting k-nearest minority neighbors. Reduces overfitting compared to simple oversampling by creating diverse examples. Most popular resampling technique for imbalanced classification. Essential when dealing with class imbalance that affects evaluation metrics like precision and recall.

### Random Oversampling

Randomly duplicates examples from minority class until dataset is balanced. Simple but can lead to overfitting as model sees exact same examples multiple times. Works well combined with regularization or ensemble methods. Fast and easy to implement.

### Random Undersampling

Randomly removes examples from majority class to balance dataset. Risk of losing potentially important information from majority class. Useful when majority class has redundant information or dataset is very large. Often combined with oversampling (hybrid approach).
