# Machine Learning Concepts - Quick Reference

## Overview

This page provides concise explanations of fundamental machine learning concepts for building, training, and evaluating models. Use this as a quick reference when implementing ML solutions, tuning hyperparameters, or diagnosing model performance issues.

**Related:** See [Data Science Concepts](ds_concepts.md) for data preparation, preprocessing, and transformation techniques that should be applied before model training.

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

## 2. ML Algorithm Types

*Common machine learning algorithms categorized by problem type and approach.*

### Linear Regression

Supervised learning algorithm for regression that models relationship between input features and continuous output using a linear equation: y = w₁x₁ + w₂x₂ + ... + b. Finds best-fit line/hyperplane by minimizing squared error. Simple, interpretable, fast to train. Assumes linear relationship between features and target. Used for price prediction, trend analysis, forecasting.

### Logistic Regression

Supervised learning algorithm for binary classification that predicts probability using sigmoid function: p = 1/(1 + e^(-z)). Despite name, used for classification not regression. Outputs probability between 0 and 1. Fast, interpretable, works well with linearly separable data. Used for spam detection, disease diagnosis, customer churn prediction.

### Decision Trees

Supervised learning algorithm that makes predictions by learning decision rules from features. Creates tree structure where each node represents a feature test, each branch represents an outcome, and each leaf represents a class label or value. Easy to interpret, handles non-linear relationships, no feature scaling needed. Prone to overfitting. Used for credit approval, medical diagnosis, customer segmentation.

### Random Forest

Ensemble learning algorithm that combines multiple decision trees trained on random subsets of data and features. Each tree votes, and majority vote (classification) or average (regression) determines final prediction. Reduces overfitting compared to single decision tree. Handles high-dimensional data well. Provides feature importance. Used for fraud detection, recommendation systems, feature selection.

### Gradient Boosting (XGBoost, LightGBM)

Ensemble learning algorithm that builds trees sequentially, where each tree corrects errors of previous trees. XGBoost and LightGBM are optimized implementations with regularization and efficient computation. Often achieves best performance on structured/tabular data. Requires careful tuning to avoid overfitting. Used for competitions, ranking problems, structured data prediction.

### Support Vector Machines (SVM)

Supervised learning algorithm that finds optimal hyperplane separating classes with maximum margin. Uses kernel trick to handle non-linear decision boundaries. Effective in high-dimensional spaces. Works well with clear margin of separation. Memory intensive for large datasets. Used for text classification, image recognition, bioinformatics.

### K-Means Clustering

Unsupervised learning algorithm that partitions data into K clusters by minimizing within-cluster variance. Iteratively assigns points to nearest centroid and updates centroids. Simple, fast, scalable. Requires specifying K in advance. Assumes spherical clusters. Used for customer segmentation, image compression, anomaly detection.

### K-Nearest Neighbors (KNN)

Supervised learning algorithm that classifies based on majority vote of K nearest neighbors or predicts based on their average. Non-parametric, no training phase (lazy learning). Simple to understand. Computationally expensive at prediction time. Sensitive to feature scaling and irrelevant features. Used for recommendation systems, pattern recognition, missing data imputation.

### Neural Networks (Deep Learning)

Supervised learning algorithms with multiple layers of interconnected neurons that learn hierarchical representations. Includes feedforward networks, CNNs (images), RNNs/LSTMs (sequences), Transformers (NLP). Can model complex non-linear relationships. Requires large amounts of data and computational resources. Used for computer vision, natural language processing, speech recognition, game playing.

### Naive Bayes

Supervised learning algorithm based on Bayes' theorem with "naive" assumption of feature independence. Fast to train and predict. Works well with high-dimensional data. Performs surprisingly well despite independence assumption. Used for text classification, spam filtering, sentiment analysis.

### Random Cut Forest (RCF)

Unsupervised learning algorithm for anomaly detection that builds ensemble of trees using random cuts through feature space. Assigns anomaly score based on how isolated a point is. Handles high-dimensional data efficiently. Doesn't require labeled anomalies. Used for fraud detection, system health monitoring, IoT sensor anomaly detection.

### Principal Component Analysis (PCA)

Unsupervised learning algorithm for dimensionality reduction that finds orthogonal axes (principal components) capturing maximum variance. Transforms features into uncorrelated components. Reduces feature space while preserving information. Helps with visualization and computational efficiency. Used for data compression, noise reduction, feature extraction.

---

## 3. Model Validation Techniques

*Strategies to assess model performance and prevent overfitting.*

### Train-Test Split

Divides dataset into separate training set (to fit model) and test set (to evaluate performance). Common splits: 70-30, 80-20, 90-10 depending on data size. Test set must never be used during training or hyperparameter tuning. Simple but can be unreliable with small datasets.

### Cross-Validation

Evaluates model by training on multiple different subsets of data and averaging results. Provides more reliable performance estimate than single train-test split. Uses all data for both training and validation, maximizing data efficiency. Essential for small datasets and hyperparameter tuning.

### K-Fold Cross-Validation

Divides data into k equal folds; trains k times, each time using k-1 folds for training and 1 for validation. Averages performance across all k runs for final estimate. Common k values: 5 or 10. Stratified k-fold maintains class proportions in each fold for classification.

---

## 4. Evaluation Metrics

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

Harmonic mean of precision and recall: 2 × (Precision × Recall) / (Precision + Recall). Balances precision and recall into a single metric, useful when both need to be reasonably high. Ranges from 0 to 1, with 1 being perfect. Preferred over accuracy for imbalanced datasets.

### ROC Curve (Receiver Operating Characteristic)

Graph plotting True Positive Rate (Recall) against False Positive Rate at various classification thresholds. Shows tradeoff between sensitivity and specificity across all possible thresholds. Curve closer to top-left corner indicates better performance. Useful for selecting optimal threshold for the use case.

### AUC (Area Under the ROC Curve)

Single number (0 to 1) summarizing ROC curve performance; measures probability that model ranks random positive higher than random negative. AUC of 0.5 means random guessing; 1.0 means perfect classification. Threshold-independent metric, useful for comparing models. Robust to class imbalance.

### AUC-PR (Area Under the Precision-Recall Curve)

Single number (0 to 1) summarizing the precision-recall curve; measures model performance across all classification thresholds by plotting precision against recall. Unlike AUC-ROC, AUC-PR focuses on the positive class performance and is more informative for highly imbalanced datasets where the minority class is of primary interest. Higher AUC-PR indicates better model performance on the positive class. Preferred over AUC-ROC when positive class is rare (fraud detection, rare disease diagnosis, anomaly detection).

---

## 5. Loss Functions

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

## 6. Optimization Algorithms

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

## 7. Training Hyperparameters

*Key parameters that control how models learn from data.*

### Batch Size

The number of training examples used in one iteration to update model weights. Smaller batches (1-32) provide noisy but frequent updates; larger batches (128-512) provide stable but less frequent updates. Affects training speed, memory usage, and convergence quality. Common values: 32, 64, 128, 256.

### Mini-batch

A subset of the training data, larger than one example but smaller than the full dataset, used for gradient descent. Combines benefits of stochastic (fast, noisy updates) and batch (stable, accurate) gradient descent. Most common approach in modern deep learning. Typically ranges from 16 to 512 examples.

### Learning Rate

Controls the step size when updating model weights during training; determines how quickly the model adapts to the problem. Too high causes unstable training or divergence; too low causes slow convergence or getting stuck. Often the most important hyperparameter to tune. Typical starting values: 0.001 to 0.1.

---

## 8. Model Performance Issues

*Understanding when the model learns too much, too little, or just right.*

### Bias-Variance Tradeoff

Bias is error from wrong assumptions (underfitting); variance is error from sensitivity to training data fluctuations (overfitting). Models with high bias are too simple; high variance models are too complex. Optimal model minimizes total error = bias² + variance + irreducible error. Core challenge in model selection.

### Overfitting

When a model learns training data too well, including its noise and peculiarities, causing poor performance on new data. The model memorizes rather than generalizes. Signs include high training accuracy but low validation accuracy. Solution: Use regularization, more data, or simpler models.

### Underfitting

When a model is too simple to capture underlying patterns in the data, resulting in poor performance on both training and test data. The model lacks the capacity to learn the relationship between features and targets. Signs include low accuracy on both training and validation sets. Solution: Use more complex models, add features, or train longer.

---

## 9. Regularization Techniques

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

## 10. Advanced ML Techniques

*Specialized machine learning approaches for specific use cases.*

### Transfer Learning

Technique that leverages knowledge from a pre-trained model (trained on large dataset) and adapts it to a new, related task with limited data. Instead of training from scratch, reuse learned features from the source task. Common approach: freeze early layers (general features like edges, textures) and fine-tune later layers (task-specific features). Dramatically reduces training time, data requirements, and computational costs.

When to Use:

- Limited labeled data for target task
- Target task is similar to source task (e.g., both are image classification)
- Want to leverage state-of-the-art pretrained models

Common Approaches:

- Feature Extraction: Freeze all pretrained layers, only train new classifier on top
- Fine-tuning: Unfreeze some layers and retrain them with small learning rate
- Progressive Unfreezing: Gradually unfreeze layers from top to bottom during training

Popular Pretrained Models:

- Vision: ResNet, VGG, EfficientNet, Vision Transformer (ViT)
- NLP: BERT, GPT, T5, RoBERTa
- Multi-modal: CLIP, Flamingo

Example Workflow:

1. Start with model pretrained on ImageNet (1.4M images, 1000 classes)
2. Remove final classification layer
3. Add new layer for specific classes (e.g., 10 classes)
4. Freeze pretrained layers, train only new layer
5. Optionally fine-tune top layers with low learning rate

Benefits:

- Requires 10-100x less data than training from scratch
- Converges faster (hours vs days/weeks)
- Often achieves better performance, especially with limited data
- Reduces computational costs significantly

GCP Implementation:
Vertex AI AutoML uses transfer learning automatically with Google's pretrained models. For custom training, TensorFlow Hub and Hugging Face provide pretrained models.

### Collaborative Filtering

Recommendation technique that makes predictions based on preferences of similar users or items. User-based finds users with similar tastes; item-based finds similar items. Doesn't require explicit feature engineering, works from interaction patterns (ratings, purchases, clicks). Used by Netflix, Amazon, and Spotify for recommendations.
