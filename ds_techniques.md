# Data Science Techniques: Quick Decision Reference

## Overview

This document helps you quickly decide which technique to apply when facing common data science challenges. Each section includes the problem symptoms, when to use the technique, intuition behind it, and practical examples.

---

## 1. Downsampling / Upsampling

**What it addresses:** Class imbalance in classification problems

### When to Downsample

**Symptoms:**

- You have 95% class A, 5% class B
- Model predicts majority class for everything
- High accuracy but poor performance on minority class

**Use when:**

- Majority class has sufficient samples even after reduction (>10,000 samples)
- Training time is a constraint
- You want to balance class representation quickly

**Intuition:** Remove excess majority class samples so the model sees balanced examples during training.

**Example:**

- Original: 95,000 negative reviews, 5,000 positive reviews
- Action: Randomly sample 5,000 negative reviews
- Result: 5,000 negative, 5,000 positive (balanced dataset)

### When to Upsample

**Use when:**

- Minority class has very few samples (<1,000)
- You can't afford to lose majority class information
- Combined with techniques like SMOTE for synthetic samples

**Intuition:** Create copies or synthetic versions of minority class to match majority class size.

**Example:**

- Original: 10,000 fraud cases, 990,000 legitimate transactions
- Action: Use SMOTE to create synthetic fraud cases
- Result: 990,000 of each class

---

## 2. Learning Rate Adjustments

**What it addresses:** Model convergence speed and training stability

### When to Decrease Learning Rate

**Symptoms:**

- Loss oscillates wildly and doesn't decrease
- Training loss jumps around or diverges
- Model parameters become NaN

**Use when:**

- Near the end of training (use learning rate schedulers)
- Training is unstable
- You've found a good region but need fine-tuning

**Intuition:** Smaller steps help you settle into a minimum without overshooting.

**Example:**

- Starting LR: 0.01 → Loss jumps between 0.5 and 2.0
- Reduced LR: 0.001 → Loss smoothly decreases from 0.5 to 0.3
- Typical values: 0.1 → 0.01 → 0.001 → 0.0001

### When to Increase Learning Rate

**Symptoms:**

- Training progresses extremely slowly
- Loss decreases by tiny amounts each epoch
- Stuck in a plateau

**Use when:**

- Early in training with a large dataset
- Model is underfitting
- Using learning rate warmup

**Intuition:** Bigger steps help you move faster through the parameter space.

**Example:**

- Starting LR: 0.00001 → 1000 epochs to reach decent performance
- Increased LR: 0.001 → 50 epochs to reach same performance

---

## 3. Dropout Adjustments

**What it addresses:** Overfitting by randomly disabling neurons during training

### When to Add/Increase Dropout

**Symptoms:**

- Training accuracy 99%, validation accuracy 75%
- Large gap between training and validation loss
- Model memorizes training data

**Use when:**

- You have limited training data
- Model is overfitting
- Network has many parameters

**Intuition:** Randomly "dropping" neurons forces the network to learn redundant representations and not rely on specific neurons.

**Example:**

- Before: No dropout → Train acc: 98%, Val acc: 72%
- After: Add dropout=0.3 → Train acc: 90%, Val acc: 85%
- Typical dropout rates: 0.2 to 0.5

### When to Decrease/Remove Dropout

**Symptoms:**

- Both training and validation accuracy are low
- Model is underfitting
- Training is very slow to converge

**Use when:**

- You have abundant training data
- Model capacity is already limited
- Underfitting is the problem

**Example:**

- Before: dropout=0.5 → Train acc: 65%, Val acc: 63%
- After: dropout=0.2 → Train acc: 82%, Val acc: 80%

---

## 4. Data Imputation

**What it addresses:** Missing values in your dataset

### When to Use Mean/Median Imputation

**Use when:**

- Data is Missing Completely at Random (MCAR)
- <5% of values are missing
- Feature is numerical and roughly normally distributed
- You need a quick, simple solution

**Intuition:** Replace missing values with the "typical" value for that feature.

**Example:**

```
Age column: [25, 30, NaN, 35, NaN, 28]
Mean imputation: [25, 30, 31.5, 35, 31.5, 28]
Median imputation: [25, 30, 29, 35, 29, 28]
```

### When to Use Forward/Backward Fill

**Use when:**

- Data is time-series or sequential
- Missing values should inherit from nearby values
- Data has temporal dependencies

**Example:**

```
Stock prices: [100, 102, NaN, NaN, 108]
Forward fill: [100, 102, 102, 102, 108]
```

### When to Use Predictive Imputation

**Use when:**

- >10% of data is missing
- Missing data has patterns (Missing at Random - MAR)
- You have computational resources
- Feature is important for your model

**Intuition:** Use other features to predict what the missing value should be.

**Example:**

- Predict missing Age using: Gender, Income, Occupation
- Train a model on complete cases, predict missing cases

### When to Create Missing Indicator

**Use when:**

- Missingness itself is informative
- Missing Not at Random (MNAR)

**Example:**

```
Income column: Many high earners leave it blank
Create: income_missing = [0, 0, 1, 0, 1]
```

---

## 5. Dimensionality Reduction

**What it addresses:** Too many features causing computational or statistical problems

### When to Use PCA (Principal Component Analysis)

**Symptoms:**

- Hundreds or thousands of features
- Features are highly correlated
- Training is extremely slow
- Multicollinearity in linear models

**Use when:**

- Features are numerical and continuous
- You want to reduce features while preserving variance
- Interpretability is not critical
- You want to visualize high-dimensional data

**Intuition:** Find new axes that capture the most variance in your data, discard axes with little variance.

**Example:**

- Original: 100 features → Model trains in 10 minutes
- After PCA: 20 components (capturing 95% variance) → Trains in 1 minute
- Use case: Image data (784 pixels) → 50 principal components

### When to Use Feature Selection

**Use when:**

- You need interpretable features
- Features are a mix of types (categorical, numerical)
- Domain knowledge is important
- You want to understand feature importance

**Methods:**

- Filter methods: Correlation, chi-square (fast, pre-modeling)
- Wrapper methods: Recursive Feature Elimination (slow, accurate)
- Embedded methods: L1 regularization/Lasso (automatic)

**Example:**

- Original: 50 features
- After correlation filter: Remove 15 highly correlated features → 35 features
- After RFE: Keep top 20 most predictive features

### When to Use t-SNE or UMAP

**Use when:**

- You want to visualize high-dimensional data in 2D/3D
- Exploring cluster structure
- NOT for model training (only visualization)

**Example:**

- Use case: Visualize 768-dimensional text embeddings in 2D to see topic clusters

---

## 6. Addressing Overfitting

**What it is:** Model performs well on training data but poorly on unseen data

**Symptoms:**

- Training accuracy >> Validation accuracy (e.g., 95% vs 70%)
- Training loss keeps decreasing, validation loss increases
- Model has learned noise instead of signal

### Solutions (in order of ease)

#### 1. Get More Training Data

- Best solution when possible
- Before: 1,000 samples → Overfitting
- After: 10,000 samples → Better generalization

#### 2. Simplify the Model

Reduce model complexity:

- Before: 5 hidden layers, 500 neurons each
- After: 2 hidden layers, 100 neurons each

or

- Before: Decision tree depth=20
- After: Decision tree depth=5

#### 3. Add Regularization

**L1 (Lasso):** Pushes some weights to exactly zero

- Use when: You want feature selection
- Example: Ridge regression with alpha=0.1

**L2 (Ridge):** Penalizes large weights

- Use when: You want to keep all features but constrain them
- Example: Linear regression with penalty='l2', C=1.0

#### 4. Increase Dropout

- Before: dropout=0.2
- After: dropout=0.4

#### 5. Early Stopping

- Stop training when validation loss stops improving
- Monitor validation loss every epoch
- Stop if no improvement for 10 consecutive epochs

#### 6. Data Augmentation (Images/Text)

- Images: Rotate, flip, crop, adjust brightness
- Text: Synonym replacement, back-translation

---

## 7. Addressing Underfitting

**What it is:** Model performs poorly on both training and validation data

**Symptoms:**

- Both training and validation accuracy are low (e.g., 65% and 63%)
- Loss remains high and plateaus quickly
- Model is too simple to capture patterns

### Solutions

#### 1. Increase Model Complexity

- Before: Logistic regression
- After: Random forest or neural network

or

- Before: 1 hidden layer, 10 neurons
- After: 3 hidden layers, 100 neurons each

#### 2. Add More Features

Create new features or use feature engineering:

**Example:** For house prices

- Original: [bedrooms, bathrooms]
- Enhanced: [bedrooms, bathrooms, bedrooms*bathrooms, total_sqft/bedrooms, age_of_house]

#### 3. Reduce Regularization

- Before: alpha=10 (very strong penalty)
- After: alpha=0.1 (weaker penalty)

#### 4. Remove/Reduce Dropout

- Before: dropout=0.5
- After: dropout=0.1 or remove dropout

#### 5. Train Longer

- Before: 10 epochs
- After: 100 epochs (if loss is still decreasing)

#### 6. Increase Learning Rate

- Before: lr=0.0001 (too slow)
- After: lr=0.001 (faster learning)

---

## 8. Improving Recall (Sensitivity)

**What it is:** Proportion of actual positives correctly identified

**Use when:** Cost of False Negatives is high (missing cancer, failing to detect fraud)

### Symptoms of Low Recall

- Model misses too many positive cases
- High precision but low recall
- False negatives are costly

### Solutions

#### 1. Lower Classification Threshold

- Before: Threshold=0.5 → Recall=60%
- After: Threshold=0.3 → Recall=85% (more liberal predictions)

#### 2. Adjust Class Weights

```python
class_weight = {0: 1, 1: 5}  # Penalize missing class 1 more
```

#### 3. Upsample Minority Class

Balance the training data to see more positive examples

#### 4. Use Recall-Oriented Metrics During Training

Optimize for F2-score (weighs recall higher than precision)

#### 5. Ensemble Methods

Use voting classifier with multiple models to catch more positives

**Example:**

Medical diagnosis scenario:

- Before: Recall=65% → Missing 35% of disease cases
- After lowering threshold: Recall=90% → Missing only 10% of disease cases
- Trade-off: Precision may decrease (more false alarms)

---

## 9. Improving Precision

**What it is:** Proportion of positive predictions that are actually correct

**Use when:** Cost of False Positives is high (spam filters, recommending bad products)

### Symptoms of Low Precision

- Too many false alarms
- High recall but low precision
- Users lose trust due to false positives

### Solutions

#### 1. Increase Classification Threshold

- Before: Threshold=0.5 → Precision=60%
- After: Threshold=0.7 → Precision=85% (more conservative predictions)

#### 2. Add More Discriminative Features

Add features that better separate positive from negative cases

#### 3. Use Ensemble Methods with Voting

Only predict positive if multiple models agree

#### 4. Better Feature Engineering

Create features that specifically identify true positives

#### 5. Clean Training Data

Remove mislabeled positive examples

**Example:**

Email spam filter:

- Before: Precision=70% → 30% of emails marked as spam are legitimate
- After raising threshold: Precision=95% → Only 5% false positives
- Trade-off: Recall may decrease (some spam gets through)

---

## 10. Quick Decision Flowchart

### Problem: Model Accuracy is Low

- **Training AND validation low?** → UNDERFITTING (increase complexity, add features, reduce regularization)
- **Training high, validation low?** → OVERFITTING (add data, simplify model, add regularization)

### Problem: Class Imbalance

- Majority class >10x minority? → Downsample majority or Upsample minority
- Use class weights in model training

### Problem: Too Many Features

- Need interpretability? → Feature selection (RFE, Lasso)
- Don't need interpretability? → PCA
- Just visualizing? → t-SNE/UMAP

### Problem: Missing Data

- <5% missing, numerical? → Mean/median imputation
- Time-series? → Forward/backward fill
- >10% missing? → Predictive imputation
- Missingness is informative? → Add missing indicator

### Problem: Training is Unstable

- Loss oscillating? → Decrease learning rate
- Loss barely moving? → Increase learning rate

### Problem: Need Higher Recall

- Lower classification threshold
- Increase class weights for positive class
- Upsample minority class

### Problem: Need Higher Precision

- Raise classification threshold
- Improve feature engineering
- Use ensemble voting (require agreement)
