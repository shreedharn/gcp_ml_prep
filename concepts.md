# Machine Learning Concepts - Quick Reference

## Overview

This page provides concise explanations of fundamental machine learning concepts organized by category. Use this as a quick reference when building models, tuning hyperparameters, or evaluating performance.

---

## 1. Core ML Concepts

*Fundamental principles underlying machine learning.*

### Supervised Learning

Learning paradigm where model trains on labeled data (input-output pairs) to learn mapping from inputs to outputs. Includes classification (discrete outputs) and regression (continuous outputs). Requires labeled training data. Examples: predicting house prices, image classification, spam detection.

### Unsupervised Learning

Learning paradigm where model finds patterns in unlabeled data without explicit target variables. Includes clustering, dimensionality reduction, and anomaly detection. No right or wrong answers, discovers hidden structure. Examples: customer segmentation, topic modeling, compression.

### Reinforcement Learning

Learning paradigm where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward. The agent learns through trial and error, receiving feedback as rewards or penalties. No labeled training data; instead learns optimal behavior through interaction. Applications include game playing, robotics, and autonomous systems.

---

## 2. Model Performance Issues

*Understanding when your model learns too much, too little, or just right.*

### Bias-Variance Tradeoff

Bias is error from wrong assumptions (underfitting); variance is error from sensitivity to training data fluctuations (overfitting). Models with high bias are too simple; high variance models are too complex. Optimal model minimizes total error = bias² + variance + irreducible error. Core challenge in model selection.

### Overfitting

When a model learns training data too well, including its noise and peculiarities, causing poor performance on new data. The model memorizes rather than generalizes. Signs include high training accuracy but low validation accuracy. Solution: Use regularization, more data, or simpler models.

### Underfitting

When a model is too simple to capture underlying patterns in the data, resulting in poor performance on both training and test data. The model lacks the capacity to learn the relationship between features and targets. Signs include low accuracy on both training and validation sets. Solution: Use more complex models, add features, or train longer.

---

## 3. Regularization Techniques

*Methods to prevent overfitting by constraining model complexity.*

### L1 Regularization (Lasso)

Adds penalty equal to absolute value of coefficient magnitudes to the loss function, encouraging sparsity by driving some weights to exactly zero. Useful for feature selection as it automatically eliminates less important features. The penalty term is λ∑|w|, where λ controls regularization strength. Creates sparse models that are easier to interpret.

### L2 Regularization (Ridge)

Adds penalty equal to square of coefficient magnitudes to the loss function, discouraging large weights but keeping all features. Distributes weights more evenly across features rather than eliminating them. The penalty term is λ∑w², where λ controls regularization strength. Preferred when all features are potentially relevant.

### Dropout

Randomly ignores (drops) a percentage of neurons during each training iteration in neural networks. Forces network to learn redundant representations, preventing over-reliance on specific neurons. Typical dropout rates: 0.2-0.5. Only applied during training, not inference.

### Early Stopping

Stops training when validation performance stops improving for specified number of epochs (patience). Prevents overfitting by avoiding unnecessary training iterations. Monitors validation loss and saves best model. Simple yet highly effective regularization technique.

---

## 4. Training Hyperparameters

*Key parameters that control how models learn from data.*

### Batch Size

The number of training examples used in one iteration to update model weights. Smaller batches (1-32) provide noisy but frequent updates; larger batches (128-512) provide stable but less frequent updates. Affects training speed, memory usage, and convergence quality. Common values: 32, 64, 128, 256.

### Mini-batch

A subset of the training data, larger than one example but smaller than the full dataset, used for gradient descent. Combines benefits of stochastic (fast, noisy updates) and batch (stable, accurate) gradient descent. Most common approach in modern deep learning. Typically ranges from 16 to 512 examples.

### Learning Rate

Controls the step size when updating model weights during training; determines how quickly the model adapts to the problem. Too high causes unstable training or divergence; too low causes slow convergence or getting stuck. Often the most important hyperparameter to tune. Typical starting values: 0.001 to 0.1.

---

## 5. Optimization Algorithms

*Methods for updating model weights during training.*

### SGD (Stochastic Gradient Descent)

Updates weights using gradient computed from one random sample (or mini-batch) at a time. Faster per iteration than batch gradient descent and can escape local minima due to noise. Converges with fluctuations; learning rate scheduling often needed. Foundation for most modern optimizers.

### Momentum

Enhancement to SGD that adds fraction of previous update to current update, helping accelerate in consistent directions. Reduces oscillations and speeds up convergence by building velocity in gradient direction. Typical momentum parameter: 0.9. Think of it as a ball rolling downhill gaining speed.

### Adam (Adaptive Moment Estimation)

Combines momentum and adaptive learning rates; maintains per-parameter learning rates adapted based on gradient history. Computes adaptive learning rates from first (mean) and second (variance) moments of gradients. Widely used default optimizer; often works well with minimal tuning. Typical hyperparameters: β₁=0.9, β₂=0.999.

### RMSprop (Root Mean Square Propagation)

Adapts learning rate for each parameter based on recent gradient magnitudes using moving average of squared gradients. Divides learning rate by root of this average, preventing oscillations in steep directions. Effective for recurrent neural networks and non-stationary problems. Developed by Geoffrey Hinton.

---

## 6. Model Validation Techniques

*Strategies to assess model performance and prevent overfitting.*

### Train-Test Split

Divides dataset into separate training set (to fit model) and test set (to evaluate performance). Common splits: 70-30, 80-20, 90-10 depending on data size. Test set must never be used during training or hyperparameter tuning. Simple but can be unreliable with small datasets.

### Cross-Validation

Evaluates model by training on multiple different subsets of data and averaging results. Provides more reliable performance estimate than single train-test split. Uses all data for both training and validation, maximizing data efficiency. Essential for small datasets and hyperparameter tuning.

### K-Fold Cross-Validation

Divides data into k equal folds; trains k times, each time using k-1 folds for training and 1 for validation. Averages performance across all k runs for final estimate. Common k values: 5 or 10. Stratified k-fold maintains class proportions in each fold for classification.

---

## 7. Evaluation Metrics

*Measures to assess model performance, especially for classification tasks.*

### Confusion Matrix

A table showing counts of True Positives, True Negatives, False Positives, and False Negatives for classification models. Rows represent actual classes, columns represent predicted classes. Provides complete picture of classification performance beyond simple accuracy. Foundation for calculating precision, recall, F1, and other metrics.

### Accuracy

The proportion of correct predictions out of total predictions: (TP + TN) / (TP + TN + FP + FN). Simple metric but can be misleading with imbalanced datasets. A model predicting all negatives on 95% negative data achieves 95% accuracy but is useless. Best used when classes are balanced and errors are equally costly.

### Precision

The proportion of positive predictions that are actually correct: True Positives / (True Positives + False Positives). Answers "Of all items we labeled as positive, how many were truly positive?" High precision means low false positive rate. Important when false positives are costly (e.g., spam detection, medical diagnosis).

### Recall (Sensitivity)

The proportion of actual positives that were correctly identified: True Positives / (True Positives + False Negatives). Answers "Of all actual positive items, how many did we correctly identify?" High recall means low false negative rate. Important when missing positives is costly (e.g., cancer detection, fraud detection).

### F1 Score

Harmonic mean of precision and recall: 2 × (Precision × Recall) / (Precision + Recall). Balances precision and recall into a single metric, useful when you need both to be reasonably high. Ranges from 0 to 1, with 1 being perfect. Preferred over accuracy for imbalanced datasets.

### ROC Curve (Receiver Operating Characteristic)

Graph plotting True Positive Rate (Recall) against False Positive Rate at various classification thresholds. Shows tradeoff between sensitivity and specificity across all possible thresholds. Curve closer to top-left corner indicates better performance. Useful for selecting optimal threshold for your use case.

### AUC (Area Under the ROC Curve)

Single number (0 to 1) summarizing ROC curve performance; measures probability that model ranks random positive higher than random negative. AUC of 0.5 means random guessing; 1.0 means perfect classification. Threshold-independent metric, useful for comparing models. Robust to class imbalance.

---

## 8. Loss Functions

*Metrics that quantify how wrong the model's predictions are.*

### Cross Entropy

Loss function that measures difference between predicted probability distribution and actual distribution, commonly used for classification. For binary classification: -[y log(p) + (1-y) log(1-p)], where y is true label and p is predicted probability. Lower values indicate better predictions; penalizes confident wrong predictions heavily. Also called log loss; widely used in neural networks.

---

## 9. Advanced ML Techniques

*Specialized machine learning approaches for specific use cases.*

### Collaborative Filtering

Recommendation technique that makes predictions based on preferences of similar users or items. User-based finds users with similar tastes; item-based finds similar items. Doesn't require explicit feature engineering, works from interaction patterns (ratings, purchases, clicks). Used by Netflix, Amazon, and Spotify for recommendations.

---

## 10. Feature Preprocessing

*Transformations applied to features before model training.*

### Feature Scaling (Normalization)

Scales features to a fixed range, typically [0, 1] using: (x - min) / (max - min). Preserves original distribution shape and relationships. Sensitive to outliers as they determine min/max. Required for distance-based algorithms like KNN, SVM, neural networks.

### Standardization (Z-score Normalization)

Transforms features to have mean=0 and standard deviation=1 using: (x - mean) / std. Assumes roughly normal distribution; less sensitive to outliers than min-max scaling. Results in unbounded range (typically -3 to +3 for normal data). Preferred for algorithms assuming normally distributed data.

---

## 11. Data Encoding Techniques

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

## 12. Missing Data Imputation

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

## 13. Sampling Techniques for Imbalanced Data

*Methods to address class imbalance in classification problems.*

### SMOTE (Synthetic Minority Over-sampling)

Creates synthetic samples for minority class by interpolating between existing minority samples and their nearest neighbors. Generates new points along line segments connecting k-nearest minority neighbors. Reduces overfitting compared to simple oversampling by creating diverse examples. Most popular resampling technique for imbalanced classification.

### Random Oversampling

Randomly duplicates examples from minority class until dataset is balanced. Simple but can lead to overfitting as model sees exact same examples multiple times. Works well combined with regularization or ensemble methods. Fast and easy to implement.

### Random Undersampling

Randomly removes examples from majority class to balance dataset. Risk of losing potentially important information from majority class. Useful when majority class has redundant information or dataset is very large. Often combined with oversampling (hybrid approach).

---

## 14. GCP ML Services & Concepts

*Google Cloud Platform specific machine learning implementations.*

### Model Selection by Problem Type

| Problem Type | Algorithms | GCP Service | AWS Service |
|--------------|------------|-------------|-------------|
| Binary Classification | Logistic Regression, Random Forest, XGBoost, Neural Networks | Vertex AI AutoML, BigQuery ML | SageMaker Autopilot |
| Multi-class Classification | Softmax Regression, Random Forest, Neural Networks | Vertex AI AutoML, BigQuery ML | SageMaker Built-in Algorithms |
| Regression | Linear Regression, Random Forest, XGBoost, Neural Networks | Vertex AI AutoML, BigQuery ML | SageMaker Autopilot |
| Time Series | ARIMA, Prophet, LSTM | BigQuery ML ARIMA_PLUS, Vertex AI | Amazon Forecast, SageMaker |
| Clustering | K-Means, DBSCAN | BigQuery ML K-Means | SageMaker K-Means |
| Recommendation | Matrix Factorization, Neural CF | BigQuery ML, Retail API | Amazon Personalize |

### BigQuery ML Evaluation Metrics

For classification models in BigQuery ML:

```sql
SELECT *
FROM ML.EVALUATE(MODEL `project.dataset.classification_model`)

-- Returns:
-- - precision: TP / (TP + FP)
-- - recall: TP / (TP + FN)
-- - accuracy: (TP + TN) / Total
-- - f1_score: 2 * (precision * recall) / (precision + recall)
-- - log_loss: -mean(y * log(p) + (1-y) * log(1-p))
-- - roc_auc: Area under ROC curve
```

### Bias and Fairness Detection

Vertex AI provides tools to analyze fairness metrics by demographic groups:

```sql
-- Check for disparate impact
SELECT
  gender,
  race,
  COUNT(*) as total,
  AVG(CAST(prediction AS FLOAT64)) as approval_rate,
  STDDEV(CAST(prediction AS FLOAT64)) as approval_stddev
FROM `project.dataset.predictions`
GROUP BY gender, race

-- Disparate Impact Ratio = (Approval Rate for Protected Group) / (Approval Rate for Reference Group)
-- Should be > 0.8 to avoid discrimination
```

**Key Metrics:**

- Demographic parity
- Equal opportunity
- Equalized odds

### Explainability with Vertex AI

Integrated Gradients (default for neural networks):

```python
from google.cloud import aiplatform

# Configure explanations
explanation_metadata = {
    'inputs': {
        'features': {
            'input_tensor_name': 'input_1',
            'encoding': 'IDENTITY',
            'modality': 'numeric',
            'index_feature_mapping': ['age', 'income', 'credit_score']
        }
    }
}

# Deploy with explanations
model.deploy(
    endpoint=endpoint,
    explanation_spec={
        'metadata': explanation_metadata,
        'parameters': explanation_parameters
    }
)

# Get predictions with explanations
instances = [{'age': 35, 'income': 75000, 'credit_score': 720}]
response = endpoint.explain(instances=instances)
```

### BigQuery ML Feature Importance

```sql
-- Global feature importance
SELECT *
FROM ML.FEATURE_IMPORTANCE(MODEL `project.dataset.my_model`)
ORDER BY importance_weight DESC
```

### Hyperparameter Tuning Strategies

- **Grid Search**: Exhaustive search over parameter grid
- **Random Search**: Random sampling from parameter space
- **Bayesian Optimization**: Uses previous results to inform next trials (most efficient)

**Best Practices:**

- Use log scale for learning rates
- Use linear scale for layer counts
- Enable early stopping to save compute costs
- Bayesian optimization requires fewer trials than grid search

### Training-Serving Skew Prevention

Use TensorFlow Transform (TFT) to guarantee training-serving consistency:

```python
import tensorflow_transform as tft

def preprocessing_fn(inputs):
    """Shared preprocessing function for training and serving"""
    outputs = {}

    # Normalize (uses full-pass statistics)
    outputs['normalized_feature'] = tft.scale_to_z_score(inputs['feature'])

    # Vocabulary (computed once on training data)
    outputs['categorical_encoded'] = tft.compute_and_apply_vocabulary(
        inputs['category'],
        top_k=1000
    )

    return outputs
```

**Common Causes of Skew:**

- Different preprocessing in training vs serving
- Missing features in serving
- Data drift over time
- Timezone differences
- Encoding issues

**Key Takeaways:**

- TFT guarantees training-serving consistency
- Prevention is better than detection
- Monitor for skew using training data baseline
- Use same preprocessing code for both training and serving
