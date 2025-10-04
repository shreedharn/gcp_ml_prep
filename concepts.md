# ML and Data Science Concepts in GCP Context

**Section Overview**: This section covers fundamental ML/DS concepts and their implementation in GCP services, with AWS comparisons.

## 4.1 Model Selection and Evaluation

### Algorithm Selection by Problem Type:

| Problem Type | Algorithms | GCP Service | AWS Service |
|--------------|------------|-------------|-------------|
| Binary Classification | Logistic Regression, Random Forest, XGBoost, Neural Networks | Vertex AI AutoML, BigQuery ML | SageMaker Autopilot |
| Multi-class Classification | Softmax Regression, Random Forest, Neural Networks | Vertex AI AutoML, BigQuery ML | SageMaker Built-in Algorithms |
| Regression | Linear Regression, Random Forest, XGBoost, Neural Networks | Vertex AI AutoML, BigQuery ML | SageMaker Autopilot |
| Time Series | ARIMA, Prophet, LSTM | BigQuery ML ARIMA_PLUS, Vertex AI | Amazon Forecast, SageMaker |
| Clustering | K-Means, DBSCAN | BigQuery ML K-Means | SageMaker K-Means |
| Recommendation | Matrix Factorization, Neural CF | BigQuery ML, Retail API | Amazon Personalize |

### Evaluation Metrics:

**Classification:**
```python
# BigQuery ML evaluation
SELECT *
FROM ML.EVALUATE(MODEL `project.dataset.classification_model`)

# Returns:
# - precision: TP / (TP + FP)
# - recall: TP / (TP + TP)
# - accuracy: (TP + TN) / Total
# - f1_score: 2 * (precision * recall) / (precision + recall)
# - log_loss: -mean(y * log(p) + (1-y) * log(1-p))
# - roc_auc: Area under ROC curve
```

**Key Takeaways:**

- Know which metrics to optimize for imbalanced datasets (AUC-PR > AUC-ROC)
- Understand precision vs recall tradeoffs
- Remember cross-validation prevents overfitting
- Know when to use different evaluation metrics

## 4.2 Bias and Fairness

### Detecting Bias:

**Vertex AI Fairness Indicators:**
```python
# Analyze fairness metrics by group
query = """
SELECT
  gender,
  race,
  COUNT(*) as total,
  AVG(CAST(prediction AS FLOAT64)) as approval_rate,
  STDDEV(CAST(prediction AS FLOAT64)) as approval_stddev
FROM `project.dataset.predictions`
GROUP BY gender, race
"""

# Check for disparate impact
# Disparate Impact Ratio = (Approval Rate for Protected Group) / (Approval Rate for Reference Group)
# Should be > 0.8 to avoid discrimination
```

**Key Takeaways:**

- Know metrics: demographic parity, equal opportunity, equalized odds
- Understand pre-training, in-training, and post-training mitigation
- Remember legal implications (disparate impact ratio)
- Know sensitive attributes should not be direct features

## 4.3 Explainability and Interpretability

### Vertex AI Explainable AI:

**Integrated Gradients (default for neural networks):**
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

**BigQuery ML Feature Importance:**
```sql
-- Global feature importance
SELECT *
FROM ML.FEATURE_IMPORTANCE(MODEL `project.dataset.my_model`)
ORDER BY importance_weight DESC
```

**Key Takeaways:**

- Integrated Gradients for deep learning, SHAP for trees
- Know global (all predictions) vs local (single prediction) explanations
- Understand baseline selection affects attributions
- Remember explanations add latency to predictions

## 4.4 Hyperparameter Tuning

### Tuning Strategies:

- **Grid Search**: Exhaustive search over parameter grid
- **Random Search**: Random sampling from parameter space
- **Bayesian Optimization**: Uses previous results to inform next trials

**Key Takeaways:**

- Bayesian optimization is most efficient (fewer trials needed)
- Grid search guarantees finding best in grid but expensive
- Random search good baseline, easy to parallelize
- Early stopping saves compute costs
- log scale for learning rates, linear for layer counts

## 4.5 Training-Serving Skew

### Causes:

Training-serving skew can result from:

- **Different preprocessing**: Training and serving use different transforms
- **Different features**: Missing features in serving
- **Data drift**: Distribution changes over time

**Prevention with TensorFlow Transform:**
```python
# Same preprocessing for training and serving
import tensorflow_transform as tft

def preprocessing_fn(inputs):
    """Shared preprocessing function"""
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

**Key Takeaways:**

- TFT guarantees training-serving consistency
- Skew monitoring requires training data baseline
- Common causes: timezone differences, encoding issues, missing values
- Prevention better than detection