# Key Technical Aspects

**Section Overview**: This section provides deep technical knowledge of critical topics, with detailed AWS comparisons to leverage your existing expertise.

**Learning Objectives:**

- Master Vertex AI training, deployment, and monitoring capabilities
- Understand data engineering patterns for ML
- Learn MLOps best practices on GCP
- Recognize optimal service choices for different scenarios

---

## 1. Platform Overview: Vertex AI (GCP's ML Platform)

### 1.1 Introduction to Vertex AI

**Vertex AI** is Google Cloud's unified ML platform that brings together all GCP ML services under a single interface. Launched in May 2021, it consolidates AI Platform, AutoML, and other ML tools into an integrated environment for the entire ML lifecycle.

**Core Philosophy:**

- Unified platform reducing service fragmentation
- Managed infrastructure with minimal operational overhead
- Native integration with GCP data services (BigQuery, Cloud Storage, Dataflow)
- Focus on production ML workflows and MLOps

**Key Components:**

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **Vertex AI Workbench** | Managed Jupyter notebooks | JupyterLab, pre-installed ML libraries, Git integration, executor for scheduled notebooks |
| **Vertex AI Training** | Custom model training | Pre-built containers, custom containers, distributed training, hyperparameter tuning |
| **Vertex AI Pipelines** | ML workflow orchestration | Kubeflow Pipelines, TFX integration, DAG visualization, artifact tracking |
| **Vertex AI Prediction** | Model serving | Online prediction (real-time), batch prediction, autoscaling, traffic splitting |
| **Vertex AI Feature Store** | Feature management | Online serving, offline training, feature versioning, point-in-time lookup |
| **Vertex AI Model Registry** | Model versioning | Model lineage, version control, deployment tracking, A/B testing |
| **Vertex AI Monitoring** | Model observability | Training-serving skew, prediction drift, feature attribution, explanation |
| **AutoML** | No-code ML | Tables, Vision, NLP, Video, structured data |
| **Vertex AI Experiments** | Experiment tracking | Metrics, parameters, artifacts, comparison across runs |
| **Vertex AI Metadata** | Lineage tracking | End-to-end tracking from data to deployed model |

**Architecture Overview:**

```
Data Sources (BigQuery, GCS, Databases)
           ↓
    Vertex AI Workbench (Development)
           ↓
    Vertex AI Training (Custom/AutoML)
           ↓
    Vertex AI Model Registry (Versioning)
           ↓
    Vertex AI Endpoints (Serving)
           ↓
    Vertex AI Monitoring (Observability)
```

### 1.2 SageMaker Reference (For Comparison)

**Amazon SageMaker** component reference for mapping to Vertex AI equivalents.

**Key Components:**

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **SageMaker Studio** | Integrated ML IDE | Notebooks, visual workflow, Git integration, debugger, profiler |
| **SageMaker Training** | Model training | Built-in algorithms, custom containers, distributed training, spot instances |
| **SageMaker Pipelines** | ML workflow orchestration | Python SDK, step caching, conditional execution, parameterized pipelines |
| **SageMaker Endpoints** | Model serving | Real-time inference, serverless inference, multi-model endpoints, async inference |
| **SageMaker Feature Store** | Feature management | Online/offline stores, feature groups, time-travel queries |
| **SageMaker Model Registry** | Model versioning | Model packages, approval workflows, cross-account deployment |
| **SageMaker Model Monitor** | Model observability | Data quality, model quality, bias drift, feature attribution |
| **SageMaker Autopilot** | AutoML | Automatic model selection, feature engineering, hyperparameter tuning |
| **SageMaker Experiments** | Experiment tracking | Run tracking, metric visualization, comparison |
| **SageMaker Clarify** | Bias detection | Pre/post training bias, explainability, feature importance |

### 1.3 Vertex AI vs SageMaker: Key Differences

#### High-Level Comparison

| Aspect | Vertex AI | SageMaker | Notes |
|--------|-----------|------------------|--------|
| **Platform Maturity** | Newer (2021), consolidation of existing services | More mature (2017), battle-tested | Built on proven GCP services |
| **Ease of Use** | Simpler, more unified interface | More features but steeper learning curve | Streamlined experience |
| **Integration** | Seamless with GCP (BigQuery, Dataflow) | Deep AWS ecosystem integration | Focus on BigQuery integration patterns |
| **AutoML Capabilities** | Strong, built-in AutoML Tables, Vision, NLP | Autopilot for structured data | Broader AutoML coverage |
| **Notebook Experience** | Workbench with managed notebooks | SageMaker Studio with more features | Less feature-rich but simpler |
| **Pipeline Orchestration** | Kubeflow Pipelines (industry standard) | SageMaker Pipelines (AWS-native) | Portable across clouds |
| **Feature Store** | Newer, simpler setup | More mature, advanced features | Simpler but fewer advanced capabilities |
| **Model Monitoring** | Built-in, automated drift detection | Comprehensive monitoring suite | More automated, less configuration |
| **Explainability** | Vertex Explainable AI | SageMaker Clarify | Similar capabilities, different APIs |

#### Detailed Feature Mapping

| Feature | Vertex AI | SageMaker |
|---------|-----------|------------------|
| **Training** | ✓ Pre-built containers<br>✓ Custom containers<br>✓ Distributed training<br>✓ Reduction server for dist. training | ✓ Built-in algorithms (18+)<br>✓ Custom containers<br>✓ Distributed training<br>✓ Spot training<br>✓ Managed warm pools |
| **Hyperparameter Tuning** | ✓ Bayesian, Grid, Random<br>✓ Parallel trials<br>✓ Early stopping | ✓ Bayesian, Grid, Random, Hyperband<br>✓ Parallel jobs<br>✓ Warm start |
| **Deployment Options** | ✓ Online prediction<br>✓ Batch prediction<br>✓ Private endpoints | ✓ Real-time endpoints<br>✓ Serverless inference<br>✓ Batch transform<br>✓ Async inference<br>✓ Multi-model endpoints |
| **Autoscaling** | ✓ CPU/GPU-based<br>✓ Min/max instances | ✓ Target tracking<br>✓ Scheduled scaling<br>✓ Application autoscaling |
| **Data Processing** | ✓ Dataflow integration<br>✓ BigQuery ML | ✓ SageMaker Processing<br>✓ Glue integration |
| **MLOps** | ✓ Vertex Pipelines<br>✓ Model Registry<br>✓ Metadata tracking | ✓ SageMaker Pipelines<br>✓ Model Registry<br>✓ Experiments<br>✓ Projects |

### 1.4 Key Advantages of Vertex AI

**Vertex AI Strengths:**

- **Unified Interface**: Less service fragmentation compared to SageMaker's many specialized tools
- **BigQuery Integration**: Direct ML on data warehouse without data movement (similar to Redshift ML but more mature)
- **AutoML Coverage**: Broader AutoML support across Tables, Vision, NLP, and Video
- **Kubeflow Pipelines**: Industry-standard orchestration (portable across clouds)
- **Simpler Pricing**: Easier cost estimation with fewer pricing dimensions

**What You'll Miss from SageMaker:**

- Serverless inference and async endpoints (Vertex AI has online/batch only)
- Multi-model endpoints (serve multiple models from single endpoint)
- Spot training for cost optimization
- More mature Feature Store with advanced capabilities
- SageMaker Studio's richer IDE experience

### 1.5 Service Name Mapping

| Functionality | Vertex AI | SageMaker |
|---------------|-----------|------------------|
| Development Environment | Vertex AI Workbench | SageMaker Studio |
| Managed Notebooks | Workbench Instances | SageMaker Notebook Instances |
| Model Training | Vertex AI Training | SageMaker Training Jobs |
| AutoML | Vertex AI AutoML | SageMaker Autopilot |
| Model Serving | Vertex AI Endpoints | SageMaker Endpoints |
| Batch Inference | Vertex AI Batch Prediction | SageMaker Batch Transform |
| Pipeline Orchestration | Vertex AI Pipelines | SageMaker Pipelines |
| Experiment Tracking | Vertex AI Experiments | SageMaker Experiments |
| Model Versioning | Vertex AI Model Registry | SageMaker Model Registry |
| Feature Management | Vertex AI Feature Store | SageMaker Feature Store |
| Model Monitoring | Vertex AI Model Monitoring | SageMaker Model Monitor |
| Explainability | Vertex Explainable AI | SageMaker Clarify |
| Bias Detection | Vertex AI (within monitoring) | SageMaker Clarify |
| Data Labeling | Vertex AI Data Labeling | SageMaker Ground Truth |

---

## 2. Deep Dive: Technical Implementation

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