# Quick Reference Tables

## GCP ML Services vs AWS Services

| GCP Service | AWS Equivalent | Primary Use Case |
|-------------|----------------|------------------|
| Vertex AI Training | SageMaker Training | Custom model training |
| Vertex AI AutoML | SageMaker Autopilot/Canvas | Automated ML |
| Vertex AI Prediction | SageMaker Endpoints | Model serving |
| Vertex AI Pipelines | SageMaker Pipelines | ML workflow orchestration |
| Vertex AI Feature Store | SageMaker Feature Store | Feature management |
| Vertex AI Model Monitoring | SageMaker Model Monitor | Drift detection |
| BigQuery ML | Redshift ML, Athena ML | SQL-based ML |
| Dataflow | AWS Glue, Kinesis Analytics | Data processing (code-based) |
| Cloud Data Fusion | AWS Glue Studio, Glue DataBrew | Data integration (visual ETL) |
| Dataproc | Amazon EMR | Spark/Hadoop workloads |
| Pub/Sub | Kinesis, SNS/SQS | Messaging |
| Cloud Storage | Amazon S3 | Object storage |
| Cloud Build | CodeBuild | CI/CD |
| Artifact Registry | ECR | Container registry |
| Cloud Composer | MWAA | Workflow orchestration |
| Vision AI | Rekognition | Image analysis |
| Natural Language AI | Comprehend | Text analysis |
| Translation API | Translate | Language translation |
| Speech-to-Text | Transcribe | Speech recognition |
| Text-to-Speech | Polly | Speech synthesis |

## When to Use Which Service

| Use Case | GCP Service | Why |
|----------|-------------|-----|
| Quick PoC with tabular data | BigQuery ML or AutoML | No ML expertise needed, SQL-based |
| Custom deep learning | Vertex AI Custom Training | Full control, TPU access |
| Large-scale batch predictions | Vertex AI Batch Prediction | Cost-effective, BigQuery integration |
| Real-time predictions | Vertex AI Endpoints | Auto-scaling, managed infrastructure |
| Streaming data processing | Dataflow | Unified batch/streaming, Apache Beam |
| Visual ETL for ML data prep | Cloud Data Fusion | No-code, 150+ connectors, business user friendly |
| Data warehouse analytics | BigQuery | Serverless, petabyte-scale |
| ML pipeline orchestration | Vertex AI Pipelines | ML-specific, metadata tracking |
| Feature engineering at scale | Dataflow + TensorFlow Transform | Training-serving consistency |
| Time series forecasting | BigQuery ML ARIMA_PLUS | SQL-based, automatic seasonality |
| Image classification (standard) | Vision AI API | Pre-trained, no training needed |
| Custom image models | Vertex AI AutoML Vision | Custom classes, transfer learning |