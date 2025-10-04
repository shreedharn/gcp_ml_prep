# Key Technical Aspects

**Section Overview**: This section provides deep technical knowledge of critical topics, with detailed AWS comparisons to leverage your existing expertise.

**Learning Objectives:**

- Master Vertex AI training, deployment, and monitoring capabilities
- Understand data engineering patterns for ML
- Learn MLOps best practices on GCP
- Recognize optimal service choices for different scenarios

## 2.1 Model Training

### Custom Training on Vertex AI

**Training Job Types:**

- Pre-built Containers: Use Google's managed containers for TensorFlow, PyTorch, scikit-learn
- Custom Containers: Bring your own Docker image for any framework
- Python Packages: Submit Python training application without containers

### Hyperparameter Tuning

**GCP - Vertex AI Hyperparameter Tuning:**
```python
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

# Define hyperparameter search space
job = aiplatform.CustomTrainingJob(
    display_name='hp-tuning-job',
    container_uri='gcr.io/my-project/trainer:latest'
)

hp_job = aiplatform.HyperparameterTuningJob(
    display_name='fraud-detection-tuning',
    custom_job=job,
    metric_spec={
        'auc': 'maximize',  # Optimization objective
    },
    parameter_spec={
        'learning_rate': hpt.DoubleParameterSpec(min=0.0001, max=0.1, scale='log'),
        'batch_size': hpt.DiscreteParameterSpec(values=[16, 32, 64, 128]),
        'num_layers': hpt.IntegerParameterSpec(min=1, max=5, scale='linear'),
        'dropout': hpt.DoubleParameterSpec(min=0.1, max=0.5, scale='linear')
    },
    max_trial_count=50,
    parallel_trial_count=5,
    search_algorithm='random',  # or 'grid', 'bayesian'
    max_failed_trial_count=10
)

hp_job.run()
```

### Distributed Training

**Strategies:**

- Data Parallelism: Replicate model across multiple workers, split data
- Model Parallelism: Split large model across multiple workers
- Reduction Server: GCP-specific feature for efficient gradient aggregation

### GPU and TPU Selection

**TPU (Tensor Processing Unit) - GCP Exclusive:**

- Custom-designed for matrix operations in neural networks
- Significantly faster for large models (BERT, ResNet, etc.)
- Cost-effective at scale
- Best for TensorFlow models

**When to Use TPUs:**

- Large batch sizes (>64)
- Models dominated by matrix multiplications
- TensorFlow-based training
- Need for maximum throughput

## 2.2 Model Deployment and Serving

### Vertex AI Prediction - Online vs Batch

**Online Prediction (Real-time):**

- Low-latency requirements (<100ms)
- Interactive applications
- Real-time fraud detection
- Recommendation systems

**Batch Prediction:**

- Large-scale inference (millions of predictions)
- Non-time-sensitive workloads
- Cost optimization (no always-on infrastructure)
- Regular scheduled predictions

### Model Versioning and Traffic Splitting

**Use Case**: Deploy new model version alongside existing version for canary or A/B testing

**Deployment Strategies:**

- Blue/Green: Deploy new version, test, then switch all traffic
- Canary: Gradually increase traffic to new version (10% → 25% → 50% → 100%)
- A/B Testing: Split traffic evenly for statistical comparison

## 2.3 Data Engineering for ML

### Vertex AI Feature Store

**Architecture:**

- Online Store: Low-latency serving for real-time predictions (milliseconds)
- Offline Store: Batch access for training data (BigQuery-based)

**Concepts:**

- Feature: Individual measurable property (e.g., age, total_purchases)
- Entity: Object being modeled (e.g., customer, product)
- Entity Type: Collection of features for an entity
- Feature Store: Container for multiple entity types

### Data Preprocessing with Dataflow and TensorFlow Transform

**TensorFlow Transform (TFT) provides:**

- Consistent preprocessing between training and serving
- Full-pass statistics over entire dataset
- TensorFlow graph-based transformations

## 2.4 ML Pipeline Orchestration

### Vertex AI Pipelines (Kubeflow Pipelines)

**Key Concepts:**

- Component: Reusable, self-contained unit of work
- Pipeline: DAG of components
- Artifact: Data produced/consumed by components
- Metadata: Lineage and execution tracking

## 2.5 Model Monitoring and Management

### Vertex AI Model Monitoring

**Types of Monitoring:**

- Training-Serving Skew: Distribution difference between training and serving data
- Prediction Drift: Change in prediction distribution over time
- Feature Attribution: Understanding which features drive predictions

**Monitoring Metrics:**

- Jensen-Shannon divergence: Measure distribution difference
- Normalized difference: Feature-level comparison
- Custom thresholds: Per-feature sensitivity

## 2.6 MLOps and CI/CD

### Continuous Training (CT) Architecture

**Components:**

- Data Monitoring: Detect data quality issues or drift
- Trigger: Initiate retraining based on schedule or drift
- Training Pipeline: Execute model training
- Validation: Evaluate new model performance
- Deployment: Deploy if better than current model

### Model Registry and Versioning:

```python
from google.cloud import aiplatform

# Register model with version and metadata
model = aiplatform.Model.upload(
    display_name='fraud-detector',
    artifact_uri='gs://my-bucket/models/fraud-detector-v1.5/',
    serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest',
    labels={
        'version': 'v1.5',
        'framework': 'scikit-learn',
        'task': 'classification',
        'team': 'fraud-ml'
    },
    description='Fraud detection model trained on 2024-01-15',
    model_id='fraud-detector-v15'
)

# Add version alias
model.add_version_aliases(['production', 'latest'])
```

**Key Takeaways:**

- Know triggers for CT: schedule, drift, data changes, manual
- Understand model versioning and aliases
- Remember approval workflows (dev → staging → production)
- Know rollback strategies