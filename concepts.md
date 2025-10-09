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

### AUC-PR (Area Under the Precision-Recall Curve)

Single number (0 to 1) summarizing the precision-recall curve; measures model performance across all classification thresholds by plotting precision against recall. Unlike AUC-ROC, AUC-PR focuses on the positive class performance and is more informative for highly imbalanced datasets where the minority class is of primary interest. Higher AUC-PR indicates better model performance on the positive class. Preferred over AUC-ROC when positive class is rare (fraud detection, rare disease diagnosis, anomaly detection).

---

## 8. Loss Functions

*Metrics that quantify how wrong the model's predictions are.*

### Mean Squared Error (MSE)

Measures average squared difference between predicted and actual values: (1/n) Σ(y - ŷ)². Most common loss function for regression problems. Heavily penalizes large errors due to squaring; sensitive to outliers. Differentiable everywhere, making it suitable for gradient-based optimization. Used in linear regression, neural networks for regression tasks. Units are squared (e.g., dollars² for price prediction).

### Mean Absolute Error (MAE)

Measures average absolute difference between predicted and actual values: (1/n) Σ|y - ŷ|. More robust to outliers than MSE since errors are not squared. Gives equal weight to all errors regardless of magnitude. Interpretable in original units (e.g., dollars for price prediction). Less sensitive to extreme values; suitable when outliers shouldn't dominate the loss.

### Huber Loss

Combines benefits of MSE and MAE; acts as MSE for small errors and MAE for large errors. Quadratic for errors below threshold δ, linear for errors above δ. More robust to outliers than MSE while maintaining differentiability. Common δ values: 1.0 or 1.35. Used in robust regression when dataset contains outliers but gradient-based optimization is needed.

### Binary Cross-Entropy (Log Loss)

Loss function for binary classification: -[y log(p) + (1-y) log(1-p)], where y is true label (0 or 1) and p is predicted probability. Heavily penalizes confident wrong predictions; small penalty for correct predictions with high confidence. Outputs range from 0 (perfect) to infinity (worst). Equivalent to negative log-likelihood for Bernoulli distribution. Standard loss for logistic regression and binary classification neural networks.

### Categorical Cross-Entropy

Extension of binary cross-entropy for multi-class classification: -Σ y_i log(p_i) across all classes. Compares one-hot encoded true labels with predicted probability distribution from softmax. Minimizing this loss is equivalent to maximizing log-likelihood. Used with softmax activation in neural networks for multi-class problems. Requires mutually exclusive classes (each sample belongs to exactly one class).

### Sparse Categorical Cross-Entropy

Functionally identical to categorical cross-entropy but accepts integer class labels instead of one-hot encoded vectors. Computationally more efficient for problems with many classes (hundreds or thousands). Example: class label is 5 instead of [0,0,0,0,0,1,0,...]. Commonly used in NLP tasks with large vocabularies and image classification with many categories.

### Hinge Loss

Loss function for maximum-margin classification, primarily used in SVMs: max(0, 1 - y·ŷ) where y ∈ {-1, 1} and ŷ is raw prediction. Encourages correct predictions to be beyond a margin; zero loss if prediction is correct and confident. Creates linear decision boundaries. Not probabilistic like cross-entropy; focuses on margin maximization. Used in SVMs and some neural network applications.

### Focal Loss

Modification of cross-entropy that down-weights easy examples and focuses on hard examples: -α(1-p)^γ log(p) for positive class. Parameter γ (typically 2) controls how much to focus on hard examples; α balances positive/negative classes. Addresses extreme class imbalance by reducing loss contribution from well-classified examples. Developed for object detection where easy negatives vastly outnumber hard positives. Particularly effective when 99%+ samples are easy negatives.

### Kullback-Leibler (KL) Divergence

Measures how one probability distribution differs from another: Σ P(x) log(P(x)/Q(x)). Asymmetric measure (KL(P||Q) ≠ KL(Q||P)); not a true distance metric. Used in variational autoencoders (VAEs) to match learned distribution to prior. Measures information loss when Q approximates P. Common in generative models and distribution matching tasks.

---

## 9. Advanced ML Techniques

*Specialized machine learning approaches for specific use cases.*

### Transfer Learning

Technique that leverages knowledge from a pre-trained model (trained on large dataset) and adapts it to a new, related task with limited data. Instead of training from scratch, reuse learned features from the source task. Common approach: freeze early layers (general features like edges, textures) and fine-tune later layers (task-specific features). Dramatically reduces training time, data requirements, and computational costs.

**When to Use:**

- Limited labeled data for target task
- Target task is similar to source task (e.g., both are image classification)
- Want to leverage state-of-the-art pretrained models

**Common Approaches:**

- **Feature Extraction**: Freeze all pretrained layers, only train new classifier on top
- **Fine-tuning**: Unfreeze some layers and retrain them with small learning rate
- **Progressive Unfreezing**: Gradually unfreeze layers from top to bottom during training

**Popular Pretrained Models:**

- **Vision**: ResNet, VGG, EfficientNet, Vision Transformer (ViT)
- **NLP**: BERT, GPT, T5, RoBERTa
- **Multi-modal**: CLIP, Flamingo

**Example Workflow:**

1. Start with model pretrained on ImageNet (1.4M images, 1000 classes)
2. Remove final classification layer
3. Add new layer for your specific classes (e.g., 10 classes)
4. Freeze pretrained layers, train only new layer
5. Optionally fine-tune top layers with low learning rate

**Benefits:**

- Requires 10-100x less data than training from scratch
- Converges faster (hours vs days/weeks)
- Often achieves better performance, especially with limited data
- Reduces computational costs significantly

**GCP Implementation:**
Vertex AI AutoML uses transfer learning automatically with Google's pretrained models. For custom training, TensorFlow Hub and Hugging Face provide pretrained models.

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

