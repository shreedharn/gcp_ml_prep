# GCP ML Services and AWS Comparables

**Section Overview**: This section maps all GCP ML services to their AWS equivalents, providing you with a comprehensive reference for service-to-service comparisons.

**Learning Objectives:**

- Identify GCP ML services and their AWS counterparts
- Understand key architectural differences between GCP and AWS approaches
- Recognize when to use each service based on use case requirements
- Master key features and integration patterns

## 1.1 Core ML Services

### Vertex AI (GCP's Unified ML Platform)

**AWS Equivalent**: Amazon SageMaker

**Service Overview**: Vertex AI is Google Cloud's unified, end-to-end ML platform that consolidates all ML services under one roof. It brings together what was previously AI Platform, AutoML, and other standalone services into a single cohesive experience.

**Key Components:**

- Vertex AI Workbench: Managed Jupyter notebooks (AWS: SageMaker Studio/Notebook Instances)
- Vertex AI Training: Custom and distributed training (AWS: SageMaker Training Jobs)
- Vertex AI Prediction: Online and batch inference (AWS: SageMaker Endpoints/Batch Transform)
- Vertex AI Pipelines: ML workflow orchestration (AWS: SageMaker Pipelines)
- Vertex AI Feature Store: Centralized feature repository (AWS: SageMaker Feature Store)
- Vertex AI Model Monitoring: Drift and skew detection (AWS: SageMaker Model Monitor)
- Vertex AI Experiments: Tracking and comparison (AWS: SageMaker Experiments)

**Architectural Differences:**

When comparing GCP and AWS ML platforms, key differences include:

- GCP: Unified console and API surface across all ML tasks
- AWS: More modular approach with separate services for different ML workflows
- GCP: Strong integration with BigQuery for data processing
- AWS: Tight integration with S3 and broader AWS data ecosystem

**Key Features:**

- Custom Training Jobs: Containerized training with support for TensorFlow, PyTorch, scikit-learn
- Hyperparameter Tuning: Automated hyperparameter optimization with configurable search strategies
- Distributed Training: Built-in support for data and model parallelism
- Model Registry: Version control and deployment management
- Metadata Management: Automatic tracking of artifacts, lineage, and experiments

**Common Use Cases:**

Vertex AI is ideal for these production ML scenarios:

- End-to-end ML workflows from data preparation to deployment
- Large-scale distributed training on TPUs or GPUs
- Production model serving with auto-scaling
- MLOps pipelines with continuous training and deployment

**Integration Points:**

- BigQuery for data warehousing and feature engineering
- Cloud Storage for dataset and model artifact storage
- Cloud Build for CI/CD integration
- Cloud Monitoring for observability

**Terminology Mapping:**

| GCP Term | AWS Term | Description |
|----------|----------|-------------|
| Training Job | Training Job | Model training execution |
| Endpoint | Endpoint | Deployed model for predictions |
| Model | Model | Trained ML model artifact |
| Pipeline | Pipeline | ML workflow orchestration |
| Custom Container | BYOC | User-provided container images |

**Practical Example - Training a Custom Model:**

**GCP (Vertex AI):**
```python
from google.cloud import aiplatform

aiplatform.init(project='my-project', location='us-central1')

# Define custom training job
job = aiplatform.CustomTrainingJob(
    display_name='fraud-detection-training',
    container_uri='gcr.io/my-project/fraud-model:latest',
    requirements=['pandas', 'scikit-learn'],
    model_serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest'
)

# Execute training
model = job.run(
    dataset=dataset,
    model_display_name='fraud-detector-v1',
    args=['--epochs=100', '--batch-size=32'],
    replica_count=1,
    machine_type='n1-standard-4',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1
)
```

**AWS (SageMaker):**
```python
import sagemaker
from sagemaker.estimator import Estimator

session = sagemaker.Session()
role = 'arn:aws:iam::123456789:role/SageMakerRole'

# Define training job
estimator = Estimator(
    image_uri='123456789.dkr.ecr.us-east-1.amazonaws.com/fraud-model:latest',
    role=role,
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    output_path='s3://my-bucket/models/',
    sagemaker_session=session
)

# Execute training
estimator.fit({'training': 's3://my-bucket/data/'})
```

**Key Differences:**

Comparing the two implementations reveals important API design differences:

- GCP uses container_uri vs AWS uses image_uri
- GCP specifies accelerator separately vs AWS includes GPU in instance type
- GCP has unified init vs AWS requires session and role management
- GCP integrates model serving container at training time

### AI Platform (Legacy) ⚠️ DEPRECATED

**Status**: Discontinued January 31, 2025. All functionality migrated to Vertex AI.

**Migration Path**: All AI Platform features are now available in Vertex AI. AI Platform is the predecessor to Vertex AI.

**Why This Matters**: You may encounter migration scenarios where knowledge of both AI Platform and Vertex AI is relevant.

### AutoML (Vertex AI AutoML)

**AWS Equivalent**: SageMaker Autopilot, SageMaker Canvas

**Service Overview**: AutoML on Vertex AI provides automated machine learning capabilities for tabular, image, text, and video data. It automates feature engineering, model selection, hyperparameter tuning, and deployment with minimal code.

**Supported Data Types:**

AutoML supports four primary data modalities:

- AutoML Tables: Structured/tabular data (classification, regression)
- AutoML Vision: Image classification, object detection
- AutoML Natural Language: Text classification, entity extraction, sentiment analysis
- AutoML Video: Video classification, object tracking

**Key Features:**

AutoML provides these automation capabilities:

- Automatic data preprocessing and feature engineering
- Neural architecture search for optimal model design
- Built-in model explainability
- One-click deployment to production

**When to Use AutoML vs Custom Training:**

**Use AutoML When:**

- Quick proof-of-concept needed
- Limited ML expertise on the team
- Standard use cases (classification, regression, common CV/NLP tasks)
- Want to establish baseline model performance
- Dataset is well-structured and labeled

**Use Custom Training When:**

- Specialized algorithms or architectures required
- Need fine-grained control over training process
- Custom loss functions or metrics needed
- Unusual data formats or preprocessing requirements
- Cost optimization through custom resource allocation

**AWS Comparison:**

| Feature | Vertex AI AutoML | SageMaker Autopilot | SageMaker Canvas |
|---------|------------------|---------------------|------------------|
| Target Users | Data scientists & developers | Data scientists | Business analysts |
| Data Types | Tabular, image, text, video | Tabular only | Tabular, image, text |
| Code Required | Minimal Python | Python API | No-code UI |
| Explainability | Built-in | Built-in | Built-in |
| Deployment | One-click | API-based | One-click |

**Practical Example - AutoML Tables:**

**GCP:**
```python
from google.cloud import aiplatform

# Create dataset
dataset = aiplatform.TabularDataset.create(
    display_name='customer-churn',
    gcs_source='gs://my-bucket/churn-data.csv'
)

# Train AutoML model
job = aiplatform.AutoMLTabularTrainingJob(
    display_name='churn-prediction',
    optimization_prediction_type='classification',
    optimization_objective='maximize-au-prc'
)

model = job.run(
    dataset=dataset,
    target_column='churned',
    training_fraction_split=0.8,
    validation_fraction_split=0.1,
    test_fraction_split=0.1,
    budget_milli_node_hours=1000,  # 1 hour
    model_display_name='churn-model-v1'
)
```

**AWS (Autopilot):**
```python
import sagemaker
from sagemaker import AutoML

session = sagemaker.Session()
automl = AutoML(
    role='arn:aws:iam::123:role/SageMakerRole',
    target_attribute_name='churned',
    output_path='s3://my-bucket/autopilot-output/',
    sagemaker_session=session
)

automl.fit(
    inputs='s3://my-bucket/churn-data.csv',
    job_name='churn-prediction',
    wait=False
)
```

### BigQuery ML

**AWS Equivalent**: Amazon Redshift ML, Amazon Athena ML

**Service Overview**: BigQuery ML enables data analysts to create and execute machine learning models using SQL queries directly within BigQuery. No need to export data or use separate ML tools.

**Supported Model Types:**

BigQuery ML supports a wide range of algorithms for different tasks:

- Linear Regression: Numeric prediction
- Logistic Regression: Binary/multiclass classification
- K-Means Clustering: Unsupervised grouping
- Matrix Factorization: Recommendation systems
- Time Series (ARIMA_PLUS): Forecasting
- Boosted Trees (XGBoost): Classification and regression
- Deep Neural Networks (DNN): Complex patterns
- AutoML Tables: Automated model selection
- Imported TensorFlow/ONNX models: Use pre-trained models

**Key Advantages:**

BigQuery ML offers significant benefits for data analysts:

- SQL-based: No Python/ML expertise required
- Data stays in place: No data movement overhead
- Scalable: Leverages BigQuery's processing power
- Fast iteration: Quick model training and evaluation

**Common Use Cases:**

Typical applications of BigQuery ML include:

- Customer segmentation with K-Means
- Churn prediction with classification models
- Product recommendations with matrix factorization
- Demand forecasting with ARIMA_PLUS
- Anomaly detection with clustering

**Practical Example - Customer Churn Prediction:**

**GCP (BigQuery ML):**
```sql
-- Create and train model
CREATE OR REPLACE MODEL `mydataset.churn_model`
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['churned'],
  auto_class_weights=TRUE,
  data_split_method='AUTO_SPLIT',
  max_iterations=50
) AS
SELECT
  customer_age,
  tenure_months,
  monthly_charges,
  total_charges,
  contract_type,
  payment_method,
  churned
FROM `mydataset.customer_data`;

-- Evaluate model
SELECT *
FROM ML.EVALUATE(MODEL `mydataset.churn_model`);

-- Make predictions
SELECT
  customer_id,
  predicted_churned,
  predicted_churned_probs[OFFSET(1)].prob as churn_probability
FROM ML.PREDICT(MODEL `mydataset.churn_model`,
  TABLE `mydataset.new_customers`
);

-- Feature importance
SELECT *
FROM ML.FEATURE_IMPORTANCE(MODEL `mydataset.churn_model`);
```

**AWS (Redshift ML):**
```sql
-- Create and train model
CREATE MODEL churn_model
FROM customer_data
TARGET churned
FUNCTION predict_churn
IAM_ROLE 'arn:aws:iam::123:role/RedshiftMLRole'
SETTINGS (
  S3_BUCKET 'my-bucket',
  MAX_RUNTIME 5000
);

-- Make predictions
SELECT
  customer_id,
  predict_churn(
    customer_age,
    tenure_months,
    monthly_charges,
    total_charges,
    contract_type,
    payment_method
  ) as predicted_churn
FROM new_customers;
```

**Key Differences:**

When comparing BigQuery ML to Redshift ML:

- BigQuery ML has richer model options (ARIMA, Matrix Factorization, DNN)
- BigQuery ML provides built-in model evaluation and feature importance
- Redshift ML relies more on SageMaker Autopilot in the background
- BigQuery ML syntax is more ML-focused, Redshift ML simpler

**Key Consideration**: Choose BigQuery ML for data analysts, SQL-based teams, and data already in BigQuery. Choose Vertex AI for complex ML workflows, Python-based teams, and when you need MLOps capabilities.

## 1.2 Pre-trained AI APIs

### Vision AI
**AWS Equivalent**: Amazon Rekognition

**Capabilities:**

Vision AI provides these pre-trained image analysis features:

- Image labeling and classification
- Face detection and recognition
- Optical Character Recognition (OCR)
- Object detection and tracking
- Explicit content detection
- Logo detection
- Landmark recognition

**When to Use**: Need quick image analysis without training custom models

### Natural Language AI
**AWS Equivalent**: Amazon Comprehend

**Capabilities:**

Natural Language AI offers these text analysis capabilities:

- Entity extraction (people, places, organizations)
- Sentiment analysis
- Syntax analysis
- Content classification
- Entity sentiment

**When to Use**: Text analysis tasks with standard requirements

### Translation API
**AWS Equivalent**: Amazon Translate

**Capabilities:**

Translation API supports:

- Text translation across 100+ languages
- Batch and real-time translation
- Custom glossaries

### Speech-to-Text
**AWS Equivalent**: Amazon Transcribe

**Capabilities:**

Speech-to-Text provides:

- Audio transcription
- Real-time streaming recognition
- Speaker diarization
- Profanity filtering
- Custom vocabulary

### Text-to-Speech
**AWS Equivalent**: Amazon Polly

**Capabilities:**

Text-to-Speech enables:

- Natural-sounding speech synthesis
- Multiple voices and languages
- Custom voice creation (in preview)
- SSML support

## 1.3 Data Processing & Storage Services

### BigQuery

**AWS Equivalent**: Amazon Redshift, Amazon Athena

**Service Overview**: BigQuery is Google's fully managed, serverless, petabyte-scale data warehouse designed for analytics and ML workloads. It's central to most GCP ML architectures.

**Key Features:**

BigQuery provides these capabilities for ML workloads:

- Serverless: No infrastructure management
- Fast queries: Parallel processing across thousands of nodes
- Streaming inserts: Real-time data ingestion
- ML integration: Native BigQuery ML support
- Cost-effective: Pay only for queries run and storage used

**Architectural Differences from Redshift:**

Comparing BigQuery to Redshift reveals fundamental design differences:

- GCP (BigQuery): Fully serverless, automatic scaling
- AWS (Redshift): Cluster-based, manual scaling required
- GCP: Separates storage and compute billing
- AWS: Combined cluster pricing

**Key Features:**

- Partitioning: Table partitioning by date/timestamp for query optimization
- Clustering: Group related data together for better performance
- Materialized Views: Pre-computed query results for faster access
- Federated Queries: Query external data sources (Cloud Storage, Cloud SQL)
- Data Transfer Service: Automated data imports from SaaS applications

**Best Practices for ML Workloads:**

- Partition training data by date for efficient querying
- Use clustering on high-cardinality columns used in WHERE clauses
- Create materialized views for frequently used feature engineering queries
- Leverage slot reservations for predictable ML pipeline costs

**Practical Example - Feature Engineering:**
```sql
-- Create partitioned table for ML features
CREATE OR REPLACE TABLE `project.dataset.customer_features`
PARTITION BY DATE(feature_timestamp)
CLUSTER BY customer_id
AS
SELECT
  customer_id,
  feature_timestamp,
  -- Aggregated features
  AVG(transaction_amount) OVER (
    PARTITION BY customer_id
    ORDER BY feature_timestamp
    ROWS BETWEEN 30 PRECEDING AND CURRENT ROW
  ) as avg_30day_spend,
  COUNT(*) OVER (
    PARTITION BY customer_id
    ORDER BY feature_timestamp
    RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND CURRENT ROW
  ) as transactions_last_7days,
  -- One-hot encoding
  IF(customer_segment = 'premium', 1, 0) as is_premium
FROM `project.dataset.transactions`;

-- Create materialized view for common features
CREATE MATERIALIZED VIEW `project.dataset.daily_customer_stats`
AS
SELECT
  DATE(transaction_time) as date,
  customer_id,
  COUNT(*) as transaction_count,
  SUM(amount) as total_spend,
  AVG(amount) as avg_transaction
FROM `project.dataset.transactions`
GROUP BY date, customer_id;
```

### Cloud Storage

**AWS Equivalent**: Amazon S3

**Service Overview**: Cloud Storage is Google's object storage service for unstructured data. It's the primary storage for datasets, model artifacts, and pipeline outputs.

**Storage Classes:**

Cloud Storage offers four storage tiers optimized for different access patterns:

- Standard: Frequently accessed data (AWS: S3 Standard)
- Nearline: Monthly access (AWS: S3 Standard-IA)
- Coldline: Quarterly access (AWS: S3 Glacier Instant Retrieval)
- Archive: Annual access (AWS: S3 Glacier Deep Archive)

**Key Features for ML:**

Cloud Storage provides these ML-specific features:

- Autoclass: Automatic lifecycle management
- Uniform bucket-level access: Simplified IAM
- Object versioning: Track model artifact versions
- Signed URLs: Temporary access for secure data sharing

**Best Practices:**

- Use Standard class for training datasets and active models
- Use Nearline/Coldline for archived experiments and old model versions
- Organize with consistent naming: gs://bucket/datasets/{train,val,test}/
- Enable versioning for model artifacts

**Example - Organizing ML Assets:**
```
gs://ml-project-bucket/
├── datasets/
│   ├── raw/
│   │   └── data-2024-01-15.csv
│   ├── processed/
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
├── models/
│   ├── fraud-detector/
│   │   ├── v1/
│   │   │   ├── model.pkl
│   │   │   └── metadata.json
│   │   └── v2/
│   └── churn-predictor/
├── experiments/
│   └── experiment-2024-01-15/
│       ├── hyperparameters.json
│       └── metrics.json
└── pipelines/
    └── training-pipeline-v1/
```

### Dataflow

**AWS Equivalent**: AWS Glue (batch), Amazon Kinesis Data Analytics/Apache Flink (streaming)

**Service Overview**: Dataflow is Google's fully managed service for stream and batch data processing based on Apache Beam. It's crucial for data preprocessing, feature engineering, and ETL in ML pipelines.

**Key Capabilities:**

Dataflow provides these powerful features:

- Unified programming: Same code for batch and streaming
- Auto-scaling: Dynamically adjust workers based on data volume
- Exactly-once processing: Guarantees for streaming data
- Windowing: Time-based aggregations for streaming data

**Common ML Use Cases:**

Dataflow is ideal for these ML scenarios:

- Large-scale data preprocessing before training
- Real-time feature computation from streaming events
- Batch prediction result processing
- Data validation with TensorFlow Data Validation (TFDV)
- Feature transformation with TensorFlow Transform (TFT)

**Architectural Pattern - Streaming Feature Engineering:**
```
Pub/Sub → Dataflow → Feature Store (online)
                  → BigQuery (offline)
```

**Practical Example - Feature Engineering Pipeline:**
```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

def compute_features(element):
    """Compute features from raw event"""
    return {
        'user_id': element['user_id'],
        'avg_session_duration': element['session_duration'] / element['page_views'],
        'days_since_last_visit': (datetime.now() - element['last_visit']).days,
        'total_purchases': element['purchase_count']
    }

# Define pipeline
options = PipelineOptions(
    runner='DataflowRunner',
    project='my-project',
    region='us-central1',
    temp_location='gs://my-bucket/temp'
)

with beam.Pipeline(options=options) as pipeline:
    (pipeline
     | 'Read from Pub/Sub' >> beam.io.ReadFromPubSub(
         subscription='projects/my-project/subscriptions/user-events')
     | 'Parse JSON' >> beam.Map(json.loads)
     | 'Compute Features' >> beam.Map(compute_features)
     | 'Write to BigQuery' >> beam.io.WriteToBigQuery(
         'my-project:dataset.user_features',
         schema='user_id:STRING,avg_session_duration:FLOAT,...',
         write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
     ))
```

### Cloud Data Fusion

**AWS Equivalent**: AWS Glue Studio, AWS Glue DataBrew

**Service Overview**: Cloud Data Fusion is a fully managed, cloud-native data integration service based on the open-source CDAP project. It provides a visual, code-free environment for building ETL/ELT pipelines, making it ideal for business users and data engineers who prefer GUI-based workflows.

**Key Capabilities:**

Data Fusion provides these features:

- Visual pipeline builder: Drag-and-drop interface for data transformations
- Pre-built connectors: 150+ connectors for databases, SaaS apps, file systems
- Data quality checks: Built-in validation and profiling
- Pipeline templates: Reusable pipeline patterns
- Hybrid/multi-cloud: Can connect on-prem and multi-cloud sources

**Data Fusion vs Dataflow:**

| Aspect | Cloud Data Fusion | Dataflow |
|--------|-------------------|----------|
| **Interface** | Visual, no-code GUI | Code-based (Apache Beam) |
| **User Persona** | Data analysts, business users | Data engineers, developers |
| **Complexity** | Simple to moderate ETL | Complex streaming/batch processing |
| **Customization** | Limited to available plugins | Full programmatic control |
| **ML Use Case** | Data preparation, cleansing | Feature engineering, real-time ML |
| **Best For** | Structured data integration | Custom ML preprocessing pipelines |

**Common ML Use Cases:**

Data Fusion is ideal for these ML scenarios:

- Consolidating data from multiple sources for ML training datasets
- Data cleansing and quality checks before model training
- Scheduled batch data ingestion from databases to BigQuery
- Simple feature engineering (aggregations, joins, filtering)
- Data preparation for AutoML or BigQuery ML

**Architectural Pattern - ML Data Preparation:**
```
Multiple Sources (DB, SaaS, Files)
           ↓
    Cloud Data Fusion (ETL)
           ↓
      BigQuery (ML Training Data)
           ↓
    Vertex AI / BigQuery ML
```

**Practical Example - Customer Data Integration Pipeline:**

Via UI (visual pipeline):
1. **Source**: MySQL database connector → customer_transactions table
2. **Transform**:
   - Joiner → Join with customer_profiles (BigQuery)
   - Group By → Aggregate purchases by customer_id
   - Wrangler → Clean null values, format dates
3. **Sink**: BigQuery → ml_datasets.customer_features

**When to Choose:**

- **Use Data Fusion when**: Simple ETL, visual interface preferred, multiple source connectors needed, business users involved
- **Use Dataflow when**: Complex transformations, streaming data, TensorFlow Transform integration, full code control needed

## 1.4 Compute Services

### Compute Engine
**AWS Equivalent**: Amazon EC2

**Use for ML:**

Compute Engine is suitable for these ML scenarios:

- Custom training environments with specific configurations
- Long-running training jobs requiring manual management
- Legacy ML code migration from on-premises

**When to Use vs Vertex AI Training:**

- Use Compute Engine: Maximum control, custom OS/software, non-standard ML frameworks
- Use Vertex AI Training: Managed ML workflows, easier scaling, built-in integrations

### Google Kubernetes Engine (GKE)
**AWS Equivalent**: Amazon EKS

**Use for ML:**

GKE excels at these ML use cases:

- Deploying custom model serving infrastructure
- Running Kubeflow for ML pipelines
- Multi-tenant ML platforms
- Batch inference jobs with complex orchestration

### Cloud Run
**AWS Equivalent**: AWS Fargate, AWS App Runner

**Use for ML:**

Cloud Run is ideal for:

- Serverless model serving
- Lightweight prediction APIs
- Batch inference jobs triggered by events
- Cost-effective for sporadic prediction workloads

## 1.5 MLOps & Development Tools

### Cloud Build
**AWS Equivalent**: AWS CodeBuild

**Use for ML:**

Cloud Build supports these ML workflows:


- Building custom training containers
- CI/CD for ML code
- Automated model retraining pipelines
- Container image creation for Vertex AI

### Artifact Registry
**AWS Equivalent**: Amazon ECR

**Use for ML:**

Artifact Registry provides:

- Store Docker images for training and serving
- Manage ML model artifacts
- Version control for containers
- Vulnerability scanning for security

## 1.6 Monitoring & Management

### Cloud Monitoring (formerly Stackdriver)
**AWS Equivalent**: Amazon CloudWatch

**Key Metrics for ML:**

Monitor these critical ML metrics:

- Training job progress and resource utilization
- Prediction latency and throughput
- Model endpoint health
- Custom metrics (model accuracy, drift scores)

### Cloud Logging
**AWS Equivalent**: Amazon CloudWatch Logs

**ML Logging Best Practices:**

Follow these logging practices for ML systems:

- Log training hyperparameters and final metrics
- Capture prediction requests for debugging
- Track data quality issues
- Monitor feature values for anomalies

## 1.7 Security & Governance

### IAM (Identity and Access Management)
**AWS Equivalent**: AWS IAM

**Key Roles for ML:**

- roles/aiplatform.user: Use Vertex AI resources
- roles/aiplatform.admin: Manage Vertex AI resources
- roles/bigquery.dataViewer: Read BigQuery data
- roles/bigquery.jobUser: Run BigQuery jobs
- roles/storage.objectViewer: Read Cloud Storage objects
- roles/ml.developer: Full ML development access

### VPC Service Controls
**AWS Equivalent**: AWS VPC Endpoints, AWS PrivateLink

**Purpose**: Create security perimeters around GCP resources to prevent data exfiltration

**ML Use Cases:**

VPC Service Controls protect ML resources:

- Protect sensitive training data in BigQuery
- Secure model artifacts in Cloud Storage
- Isolate Vertex AI resources within network perimeter

### Cloud KMS (Key Management Service)
**AWS Equivalent**: AWS KMS

**Use for ML:**

Cloud KMS secures ML data and artifacts:

- Encrypt training data at rest
- Protect model artifacts
- Encrypt BigQuery datasets
- Secure sensitive feature data