# Data Splitting Techniques for Machine Learning

## Overview

Data splitting is the process of dividing a dataset into separate subsets for training, validating, and testing machine learning models. The splitting strategy chosen significantly impacts how well a model performs in production. This guide covers the most common splitting techniques, when to use them, and practical examples.

---

## Understanding Training, Validation, and Test Datasets

Before diving into splitting techniques, it's essential to understand why data is split in the first place and what each subset is used for.

When building machine learning models, all data is not used for the same purpose. Instead, data is divided into three distinct datasets, each serving a specific role in the model development lifecycle:

- **Training dataset**: Where the model learns patterns and relationships
- **Validation dataset**: Where the model is evaluated and tuned during development
- **Test dataset**: Where an honest, final assessment of model performance is obtained

Think of it like learning a new skill: practice (training), receive feedback and adjust the approach (validation), then take a final test to assess true mastery (testing). Each dataset plays a crucial role in ensuring the model performs well on real-world data it has never seen before.

The table below provides a detailed comparison of these three datasets:

| Aspect | Training dataset | Validation dataset | Test dataset |
|--------|-----------------|-------------------|--------------|
| **Purpose** | Train the machine learning model | Tune hyperparameters and make model architecture decisions | Final, unbiased evaluation of the model |
| **What happens** | Model learns patterns, relationships, and features. Adjusts internal parameters (weights and biases) to minimize errors | Evaluate model performance to:<br>• Choose between models/algorithms<br>• Tune hyperparameters<br>• Implement early stopping<br>• Make architecture decisions | Evaluate completely unseen data once for honest assessment of production performance |
| **Student analogy** | Studying from textbooks and practice problems to learn concepts | Taking practice tests to identify weak areas and adjust study strategies | Taking the final exam once - represents real-world performance |
| **Frequency of use** | Model sees this data multiple times during training | Can be evaluated multiple times; make model changes based on results | Touch only once at the very end |
| **Model interaction** | Model learns from this data | Model is evaluated (not trained) on this data | Model is evaluated (never trained or tuned) on this data |
| **Typical size** | 60-80% of total data | 10-20% of total data | 10-20% of total data |
| **Key principle** | Used to fit model parameters | Used to select and tune the model | Used only for final performance measurement |
| **Critical rule** | Must be representative of the problem | Must never be used for training | Never use results to improve the model - prevents bias |

### Why split data?

**1. Prevent overfitting**

- Training and testing on the same data causes the model to memorize rather than learn
- The model will perform perfectly on training data but fail on new, unseen data

**2. Honest performance estimation**

- Training accuracy is optimistic (model has seen the answers)
- Test accuracy reflects real-world performance

**3. Model selection and tuning**

- Validation set allows comparison between models without biasing the final evaluation
- Prevents information leakage from the test set

### Common misconception

❌ **Wrong:** Using test data to improve the model

- This defeats the purpose - training on the test set
- Performance estimate becomes biased

✅ **Right:** Using validation data to improve the model, then using test data once for final evaluation

### The golden rule

**Never touch the test set until model development is completely done.** It's the only source of truth for how the model will perform in the real world.

---

## 1. Random split (simple holdout)

**Brief description:** Randomly divide data into training and test sets, typically 70-30 or 80-20.

**Intuition:** Every data point has an equal chance of ending up in either set, like shuffling a deck of cards and dealing them into piles.

**When to use:**

- Large datasets with independent observations
- No temporal dependencies
- Classes are roughly balanced

**Example:**

```
Dataset: 1000 samples
Random shuffle → [847, 23, 991, 445, ...]
Train: First 800 samples (80%)
Test: Last 200 samples (20%)
```

**Pros:** Simple, fast, works well for large datasets

**Cons:** Variance in results, may not represent class distribution, wastes data (only trains once)

---

## 2. Stratified split

**Brief description:** Random split that preserves the proportion of classes in both train and test sets.

**Intuition:** If 30% of the data is class A, both train and test sets will have 30% class A. Like ensuring each hand in a card game has the same ratio of suits.

**Step-by-step example:**

Original dataset (100 samples):
- class A: 70 samples (70%)
- class B: 30 samples (30%)

After 80-20 stratified split:

Training set (80 samples):
- class A: 56 samples (70%)
- class B: 24 samples (30%)

Test set (20 samples):
- class A: 14 samples (70%)
- class B: 6 samples (30%)

**When to use:**

- Imbalanced datasets
- Classification problems
- Small datasets where class distribution matters

**Pros:** Reduces variance, ensures representative test set

**Cons:** Only works for classification, slightly more complex

---

## 3. K-fold cross-validation

**Brief description:** Split data into K equal parts (folds). Train on K-1 folds, test on 1 fold, and repeat K times so each fold serves as the test set once.

**Intuition:** Everyone gets a turn being the test set. Like rotating team members through different roles.

**Step-by-step example (5-fold CV):**

Dataset: 100 samples divided into 5 folds of 20 samples each

```
Iteration 1: Train on [fold 2,3,4,5], test on [fold 1]
Iteration 2: Train on [fold 1,3,4,5], test on [fold 2]
Iteration 3: Train on [fold 1,2,4,5], test on [fold 3]
Iteration 4: Train on [fold 1,2,3,5], test on [fold 4]
Iteration 5: Train on [fold 1,2,3,4], test on [fold 5]

Final score: Average of all 5 test scores
```

**When to use:**

- Small to medium datasets
- Need robust performance estimate
- Comparing multiple models

**Pros:** Uses all data for training and testing, reduces variance, robust estimate

**Cons:** K times slower, still random (not for time series)

**Common K values:** 5 or 10

---

## 4. Stratified K-fold cross-validation

**Brief description:** K-fold CV where each fold maintains the original class distribution.

**Intuition:** Combines the benefits of stratification with cross-validation.

**Example:**

Dataset: 100 samples (70% class A, 30% class B)
5-fold stratified CV

Each fold has 20 samples:
- Fold 1: 14 class A, 6 class B
- Fold 2: 14 class A, 6 class B
- Fold 3: 14 class A, 6 class B
- Fold 4: 14 class A, 6 class B
- Fold 5: 14 class A, 6 class B

Same rotation as K-fold CV

**When to use:**

- Small, imbalanced classification datasets
- Gold standard for model evaluation on static data

---

## 5. Group/cluster split

**Brief description:** Split data ensuring all samples from the same group stay together in either train or test.

**Intuition:** When multiple photos of the same person exist, all photos of person A go into either train or test, never both. Prevents the model from memorizing individuals.

**Step-by-step example:**

Medical dataset: patient X-rays

```
Patient 1: [scan_a, scan_b, scan_c]
Patient 2: [scan_d, scan_e]
Patient 3: [scan_f, scan_g, scan_h]
Patient 4: [scan_i]

Group split (by patient):
Train: patient 1, patient 3 → [scan_a,b,c,f,g,h]
Test:  patient 2, patient 4 → [scan_d,e,i]

❌ WRONG (random): Training on scan_a,b from patient 1, testing on scan_c from patient 1
```

**When to use:**

- Multiple samples from same entity (patients, users, locations)
- Hierarchical data structures
- Prevent data leakage from correlated samples

**Pros:** Prevents overfitting to specific groups, realistic evaluation

**Cons:** May need larger datasets, unequal split sizes

---

## 6. Time series cross-validation (walk-forward/rolling window)

**Brief description:** For time series, progressively train on expanding (or fixed) windows and test on the next period.

**Intuition:** Simulate real-world deployment where retraining occurs periodically. Like using last month's data to predict next month, then updating.

**Step-by-step example (expanding window):**

Time series: [Jan, Feb, Mar, Apr, May, Jun, Jul, Aug]

```
Fold 1: Train [Jan,Feb,Mar] → test [Apr]
Fold 2: Train [Jan,Feb,Mar,Apr] → test [May]
Fold 3: Train [Jan,Feb,Mar,Apr,May] → test [Jun]
Fold 4: Train [Jan,Feb,Mar,Apr,May,Jun] → test [Jul]
Fold 5: Train [Jan,Feb,Mar,Apr,May,Jun,Jul] → test [Aug]
```

**Rolling window variant (fixed size):**

```
Fold 1: Train [Jan,Feb,Mar] → test [Apr]
Fold 2: Train [Feb,Mar,Apr] → test [May]
Fold 3: Train [Mar,Apr,May] → test [Jun]
Fold 4: Train [Apr,May,Jun] → test [Jul]
```

**When to use:**

- Time series forecasting
- Models need retraining over time
- Evaluating model degradation

---

## 7. Leave-one-out cross-validation (LOOCV)

**Brief description:** Extreme case of K-fold where K = number of samples. Train on N-1 samples, test on 1.

**Intuition:** Each individual data point gets to be the test set.

**Example:**

Dataset: 50 samples

```
Iteration 1: Train on samples [2-50], test on [1]
Iteration 2: Train on samples [1,3-50], test on [2]
...
Iteration 50: Train on samples [1-49], test on [50]
```

**When to use:**

- Very small datasets (< 100 samples)
- Every data point is precious

**Pros:** Maximum use of data

**Cons:** Very computationally expensive, high variance

---

## 8. Nested cross-validation

**Brief description:** Two-level CV: outer loop for model evaluation, inner loop for hyperparameter tuning.

**Intuition:** Prevents hyperparameter tuning from leaking information into the test set. The outer loop never sees the hyperparameter selection process.

**Step-by-step example:**

```
Outer loop (5-fold for evaluation):
  For each outer fold:
    Inner loop (3-fold for tuning):
      - Try different hyperparameters
      - Select best hyperparameter
    - Train final model with best params
    - Evaluate on outer test fold

Result: Unbiased performance estimate
```

**When to use:**

- Need both hyperparameter tuning AND unbiased evaluation
- Comparing models fairly
- Publication-quality results

**Pros:** No information leakage, proper estimate

**Cons:** Very computationally expensive

---

## 9. Train-validation-test split (three-way split)

**Brief description:** Divide data into three sets: train (60%), validation (20%), test (20%).

**Intuition:** Training builds the model, validation tunes it, test evaluates it honestly. Like practice games, scrimmages, and championship matches.

**Example:**

```
Dataset: 1000 samples

Train: 600 samples (build models)
Validation: 200 samples (tune hyperparameters, select model)
Test: 200 samples (final evaluation, touch ONCE)
```

**When to use:**

- Large datasets
- Deep learning (need validation for early stopping)
- When hyperparameters need tuning

**Workflow:**

1. Train multiple models on training set
2. Evaluate on validation set, tune hyperparameters
3. Select best model
4. Final evaluation on test set (once only)

---

## 10. Blocked/purged cross-validation

**Brief description:** For time series, add gaps between train and test to prevent temporal leakage.

**Intuition:** Real-world predictions need time to materialize. When predicting stock prices, same-day information cannot be used.

**Example:**

Time series with 2-day purge:

```
Fold 1: Train [Day 1-30] | PURGE [Day 31-32] | test [Day 33-35]
Fold 2: Train [Day 1-35] | PURGE [Day 36-37] | test [Day 38-40]

The purge period ensures no overlap between training features and test outcomes.
```

**When to use:**

- Financial time series
- Data where features take time to compute
- Preventing look-ahead bias

---

## Quick selection guide

| Scenario | Recommended technique |
|----------|----------------------|
| Large dataset, independent samples | Random split |
| Imbalanced classes | Stratified split or stratified K-fold |
| Small dataset | Stratified K-fold CV |
| Time series data | Temporal split or walk-forward CV |
| Multiple samples per entity | Group split |
| Need hyperparameter tuning | Train-validation-test or nested CV |
| Financial/trading data | Blocked/purged CV |
