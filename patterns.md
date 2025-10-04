# Architectural Patterns and Design Principles

**Section Overview**: This section covers common ML architecture patterns, with detailed implementation guidance for both GCP and AWS.

**Learning Objectives:**

- Recognize standard ML architecture patterns
- Choose appropriate services for different use cases
- Design scalable and cost-effective solutions
- Understand trade-offs between architectural approaches

## 3.1 Real-Time Prediction Architecture

**Pattern Description**: Low-latency (<100ms) prediction serving for interactive applications

**GCP Implementation:**
```
API Gateway / Cloud Load Balancer
          ↓
    Cloud Run (API)
          ↓
    Memorystore (Redis) - [Cache Layer]
          ↓
  Vertex AI Endpoint
  (Multiple models with traffic splitting)
          ↓
    Model (deployed on n1-standard-4 + GPU)
```

**AWS Implementation:**
```
API Gateway / Application Load Balancer
          ↓
    Lambda or ECS (API)
          ↓
    ElastiCache (Redis) - [Cache Layer]
          ↓
  SageMaker Endpoint
  (Multiple variants with traffic splitting)
          ↓
    Model (deployed on ml.m5.xlarge + GPU)
```

**Key Differences:**

- GCP: Cloud Run offers automatic scaling with zero-to-scale capability
- AWS: Lambda for serverless or ECS for containerized workloads
- GCP: Memorystore for Redis is fully managed
- AWS: ElastiCache requires more configuration but offers more control
- Both: Support traffic splitting and A/B testing at the endpoint level

**When to Use This Pattern:**

✅ Latency requirements <100ms
✅ High request volume with repeated queries
✅ Need global availability
✅ Variable traffic patterns

## 3.2 Batch Prediction Pipeline

**Pattern Description**: Large-scale offline inference for millions of predictions

**GCP Implementation:**
```
Cloud Scheduler
      ↓
Cloud Functions (Trigger)
      ↓
BigQuery (Input Data) → Vertex AI Batch Prediction → BigQuery (Results)
      ↓
Dataflow (Post-processing)
      ↓
BigQuery (Final Results)
```

**AWS Implementation:**
```
EventBridge (CloudWatch Events)
      ↓
Lambda (Trigger)
      ↓
Athena/Redshift (Input Data) → SageMaker Batch Transform → S3 (Results)
      ↓
AWS Glue (Post-processing)
      ↓
Redshift/Athena (Final Results)
```

**Key Differences:**

- GCP: BigQuery provides unified data warehouse and prediction input/output
- AWS: Separate services for storage (S3), querying (Athena/Redshift), and batch inference
- GCP: Dataflow offers unified batch and stream processing
- AWS: Glue for batch ETL, separate from streaming (Kinesis)
- GCP: Cloud Scheduler is dedicated service; AWS uses EventBridge/CloudWatch Events

**When to Use This Pattern:**

Batch prediction is suitable when:

✅ Millions of predictions needed
✅ Not time-sensitive (can take hours)
✅ Cost optimization important
✅ Periodic/scheduled inference

## 3.3 Streaming ML Pipeline

**Pattern Description**: Real-time feature engineering and prediction on streaming data

**GCP Implementation:**
```
Pub/Sub (Event Stream)
      ↓
Dataflow (Feature Engineering)
      ↓   ↓   ↓
      |   |   +→ BigQuery (Offline Storage)
      |   +→ Feature Store (Online)
      +→ Vertex AI Endpoint (Predictions)
           ↓
      Pub/Sub (Prediction Results)
           ↓
      Application (Actions)
```

**AWS Implementation:**
```
Kinesis Data Streams (Event Stream)
      ↓
Kinesis Data Analytics / Flink (Feature Engineering)
      ↓   ↓   ↓
      |   |   +→ S3 + Athena (Offline Storage)
      |   +→ SageMaker Feature Store (Online)
      +→ SageMaker Endpoint (Predictions)
           ↓
      Kinesis Data Streams (Prediction Results)
           ↓
      Lambda / Application (Actions)
```

**Key Differences:**

- GCP: Pub/Sub is fully managed message queue with global availability
- AWS: Kinesis Data Streams requires shard management
- GCP: Dataflow provides unified SDK (Apache Beam) for batch and stream
- AWS: Kinesis Analytics or managed Flink for stream processing
- GCP: Feature Store integrated with Vertex AI ecosystem
- AWS: SageMaker Feature Store with separate integration requirements
- GCP: Single Dataflow job can write to multiple destinations
- AWS: May need separate Lambda functions or Kinesis Firehose for multi-destination writes

**When to Use This Pattern:**

Streaming ML pipelines excel when:

✅ Real-time decision making required
✅ Continuous event streams
✅ Need for feature aggregation across time windows
✅ Low-latency requirements (seconds, not milliseconds)

## 3.4 End-to-End AutoML Workflow

**Pattern Description**: Automated workflow from raw data to deployed model

**GCP Implementation (Vertex AI Pipelines):**
```python
from kfp import dsl
from google.cloud.aiplatform import pipeline_jobs

@dsl.pipeline(name='automl-e2e-pipeline')
def automl_pipeline(
    project_id: str,
    dataset_uri: str,
    target_column: str,
    model_name: str,
    budget_hours: int = 1
):
    # Create dataset, train AutoML model, create endpoint, deploy model
    pass

# Execute
job = pipeline_jobs.PipelineJob(
    display_name='automl-churn-pipeline',
    template_path='automl_pipeline.yaml',
    parameter_values={
        'project_id': 'my-project',
        'dataset_uri': 'gs://my-bucket/churn_data.csv',
        'target_column': 'churned',
        'model_name': 'churn-predictor',
        'budget_hours': 2
    }
)

job.run()
```

**AWS Implementation (SageMaker Pipelines):**
```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CreateModelStep
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator

# Define preprocessing step
sklearn_processor = SKLearnProcessor(
    framework_version='0.23-1',
    role=sagemaker_role,
    instance_type='ml.m5.xlarge',
    instance_count=1
)

# Define AutoML training using Autopilot
from sagemaker.automl.automl import AutoML

automl = AutoML(
    role=sagemaker_role,
    target_attribute_name='churned',
    output_path='s3://my-bucket/autopilot-output',
    max_candidates=10,
    max_runtime_per_training_job_in_seconds=3600
)

# Create and execute pipeline
pipeline = Pipeline(
    name='automl-churn-pipeline',
    steps=[preprocessing_step, training_step, model_step, deployment_step]
)

pipeline.upsert(role_arn=sagemaker_role)
execution = pipeline.start()
```

**Key Differences:**

- GCP: Kubeflow Pipelines DSL with Python decorators (@dsl.pipeline)
- AWS: SageMaker Pipelines with explicit step definitions
- GCP: AutoML is integrated component in Vertex AI
- AWS: SageMaker Autopilot is separate from Pipelines, requires integration
- GCP: Vertex AI Pipelines stores artifacts in GCS automatically
- AWS: Must explicitly define S3 paths for all artifacts
- GCP: Built-in model registry integration
- AWS: Requires explicit Model Registry step configuration

## 3.5 MLOps Pipeline (CI/CD/CT)

**Pattern Description**: Continuous Integration, Deployment, and Training for ML

**GCP Complete MLOps Architecture:**
```
Code Repository (Cloud Source Repo / GitHub)
      ↓
Cloud Build (CI: Test, Build, Push)
      ↓
Artifact Registry (Container Storage)
      ↓
Vertex AI Pipelines (Training Pipeline)
      ↓
Model Registry (Version Management)
      ↓
Vertex AI Endpoints (Deployment)
      ↓
Model Monitoring (Drift Detection)
      ↓ (Trigger on Drift)
Cloud Functions → Re-trigger Training
```

**AWS Complete MLOps Architecture:**
```
Code Repository (CodeCommit / GitHub)
      ↓
CodePipeline + CodeBuild (CI: Test, Build, Push)
      ↓
ECR (Elastic Container Registry)
      ↓
SageMaker Pipelines (Training Pipeline)
      ↓
SageMaker Model Registry (Version Management)
      ↓
SageMaker Endpoints (Deployment)
      ↓
SageMaker Model Monitor (Drift Detection)
      ↓ (Trigger on Drift)
Lambda + EventBridge → Re-trigger Training
```

**Key Differences:**

- GCP: Cloud Build is unified CI/CD service with native GCP integration
- AWS: CodePipeline + CodeBuild require more configuration
- GCP: Artifact Registry for both containers and language packages
- AWS: ECR for containers, CodeArtifact for packages (separate services)
- GCP: Vertex AI provides integrated monitoring with automated triggers
- AWS: SageMaker Model Monitor requires manual EventBridge rule configuration
- GCP: Cloud Functions directly trigger Vertex AI Pipelines
- AWS: Lambda triggers SageMaker Pipelines via SDK calls
- GCP: Single Vertex AI console for entire MLOps workflow
- AWS: Multiple consoles (CodePipeline, SageMaker, CloudWatch, Lambda)

**Continuous Training Triggers:**

| Trigger Type | GCP Implementation | AWS Implementation |
|--------------|-------------------|-------------------|
| **Schedule** | Cloud Scheduler → Cloud Functions → Vertex AI Pipeline | EventBridge (cron) → Lambda → SageMaker Pipeline |
| **Drift Detection** | Vertex AI Model Monitoring → Pub/Sub → Cloud Functions | SageMaker Model Monitor → EventBridge → Lambda |
| **Data Quality** | TFDV + Cloud Functions → Vertex AI Pipeline | Glue Data Quality → EventBridge → Lambda |
| **Manual** | Vertex AI Console / gcloud CLI | SageMaker Console / AWS CLI |

**Key Takeaways:**

- Know the difference between CI (code), CD (deployment), CT (continuous training)
- Understand trigger mechanisms: schedule, drift, data quality
- Remember approval gates for production deployment
- Know rollback strategies (traffic splitting, blue/green)
- GCP offers more integrated MLOps experience; AWS provides more flexibility with separate services