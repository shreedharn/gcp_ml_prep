# Quick Reference Tables

## GCP Data and AI/ML Services

| Service / Tool | What It Provides | When to Use It | Interoperability / Related To |
|----------------|------------------|----------------|--------------------------------|
| **Vertex AI Pipelines** | Orchestrates end-to-end ML workflows (built on KFP). Reproducible DAGs from data prep → training → deploy → monitoring. | Use for any production ML workflow, automated retraining, CI/CD for ML. | Integrates with Custom Training, AutoML, Dataflow, BigQuery, TFX, Feature Store, Model Registry, Endpoints, Model Garden |
| **Vertex AI Custom Training** | Fully customizable training with custom containers or pre-built frameworks | Training custom models with TensorFlow, PyTorch, scikit-learn, XGBoost when AutoML insufficient | Pipelines, Deep Learning Containers, Cloud TPU, GPU, Hyperparameter Tuning, Experiments, Model Registry, TensorBoard |
| **Vertex AI Batch Prediction** | Asynchronous batch inference at scale | Large-scale offline predictions, BigQuery table scoring, cost-effective inference | BigQuery, Cloud Storage, Model Registry, Pipelines |
| **Vertex AI Notebooks** | Preconfigured JupyterLab environment | Rapid ML prototyping and data exploration | Workbench, Notebook Runner |
| **Vertex AI Workbench (Managed)** | Enterprise IAM/VPC-secure notebook | Long-running governed development | BigQuery, Vertex AI SDK, Cloud Storage, Git integration |
| **Vertex AI Notebook Runner** | Executes notebooks as reproducible jobs | Convert exploratory notebooks into production jobs | Pipelines, Workbench, Cloud Scheduler |
| **Vertex AI AutoML** | Automated model training | Auto feature engineering + HPO when you want fully managed ML | Managed Datasets, Model Registry, Endpoints, Pipelines |
| **AutoMLTabularTrainingJob** | AutoML Tabular via SDK | Programmatic training workflows | Managed Datasets, Pipelines, Model Registry |
| **AutoML Natural Language** | Text classification, sentiment, entity extraction | No-code or low-code text ML | Cloud Document AI, Natural Language AI API, Translation API |
| **AutoML Vision Edge** | Vision models optimized for edge | Low-latency, offline, mobile/IoT inference | AutoML Vision, Edge deployment platforms, TensorFlow Lite |
| **AutoML Edge** | General edge AutoML export | On-device inference constraints | AutoML Vision, AutoML NLP, mobile/IoT platforms |
| **AutoML Table (legacy)** | Older AutoML tabular system | Legacy workflows / migrations | BigQuery, AutoML Tabular |
| **Tabular Workflow for TabNet** | Deep learning tabular training | Need interpretable TabNet model | AutoML Tabular, Custom Training |
| **Vertex Vision Occupancy Analytics** | Prebuilt occupancy/people counting | Retail or physical space occupancy analytics | Vision AI API, AutoML Vision |
| **Vertex AI Experiments** | Track runs, metrics, parameters | Compare training runs | Custom Training, TensorBoard, Hyperparameter Tuning |
| **Vertex AI Metadata** | Lineage + artifact tracking | Compliance, auditing, reproducibility | Model Registry, Pipelines, Datasets, Artifacts |
| **Vertex AI TensorBoard** | Training visualization | DL training diagnostics | Custom Training, Experiments, Hyperparameter Tuning |
| **Vertex AI Managed Dataset** | Dataset metadata layer for versioning and lineage | Create and manage datasets for AutoML and custom training | AutoML, Custom Training, Pipelines, Cloud Storage |
| **Vertex AI Feature Store** | Online/offline feature serving with point-in-time correctness | Real-time feature delivery, train/serve consistency, feature sharing across teams | Endpoints, Batch Prediction, Bigtable, BigQuery, Pipelines |
| **Vertex AI Model Registry** | Central versioned model store with lineage and metadata | Manage multiple model versions, track model lineage, promote models across environments | Custom Training, AutoML, Endpoints, Batch Prediction, Pipelines |
| **Vertex AI Endpoint** | Real-time prediction serving with autoscaling | Low-latency online inference with traffic splitting and A/B testing | Model Registry, Monitoring, Feature Store, Matching Engine |
| **Vertex AI Model Monitoring** | Training-serving skew, prediction drift, and feature attribution drift detection | Production model safety, detect data drift and model degradation | Endpoints, Feature Store, Explainable AI, Cloud Monitoring |
| **Vertex Explainable AI** | Feature attribution methods (Sampled Shapley, XRAI, Integrated Gradients) | Model interpretability, fairness audits, regulatory compliance | Custom Training, AutoML, Endpoints, Model Monitoring |
| **Vertex AI Hyperparameter Tuning** | Distributed HPO using grid, random, or Bayesian search | Systematic search for best hyperparameters in custom training jobs | Custom Training, Experiments, Vizier, Pipelines |
| **Vertex AI Vizier** | Black-box optimization service (HPO as a service) | Optimize hyperparameters or arbitrary objectives across multiple trials | Hyperparameter Tuning, Custom Training, Experiments |
| **Vertex AI Matching Engine** | High-scale, low-latency vector similarity search | Semantic search, recommendation systems, similarity matching for embeddings | Endpoints, Feature Store, Custom embeddings, Recommendations AI |
| **Vertex AI Labeling** | Human-in-the-loop data annotation service | Create labeled training datasets with managed workforce | Managed Datasets, AutoML, Custom Training, Cloud Storage |
| **Cloud TPU** | Tensor Processing Units (ML accelerators) | Accelerate TensorFlow/JAX/PyTorch training, especially large models and matrix operations | Custom Training, Deep Learning VMs, Deep Learning Containers, TensorFlow, JAX |
| **Deep Learning VM Images** | Pre-configured VMs with ML frameworks | Quick setup for custom development environments with GPU/TPU support | Custom Training, Notebooks, Cloud TPU, GPU |
| **Deep Learning Containers** | Pre-built Docker images with ML frameworks | Containerized training/serving with TensorFlow, PyTorch, scikit-learn | Custom Training, Pipelines, Artifact Registry, Cloud Build |
| **BigQuery** | Serverless, petabyte-scale SQL data warehouse | Store and transform training data, feature engineering, batch predictions at scale | BigQuery ML, Feature Store, Batch Prediction, Dataflow, Data Fusion |
| **BigQuery ML** | Train and deploy ML models using SQL (linear regression, DNN, XGBoost, AutoML) | Warehouse-native ML for data analysts, rapid prototyping without data movement | BigQuery, Vertex AI Model Registry, Endpoints, Batch Prediction |
| **BigQuery Scheduled Queries** | Scheduled SQL execution with cron-like syntax | Automated feature refreshes, periodic batch predictions, ETL pipelines | BigQuery ML, Cloud Scheduler, Dataflow |
| **Dataflow** | Fully managed Apache Beam service for batch and streaming | Large-scale feature engineering, data preprocessing, distributed batch inference | Pipelines, TFX, TensorFlow Transform, BigQuery, Pub/Sub, Cloud Storage |
| **DataflowRunner** | Apache Beam runner for GCP Dataflow | Execute Beam pipelines on Dataflow for scalable data processing | Apache Beam, TFX, TensorFlow Transform |
| **TFRecord** | TensorFlow-optimized binary data format | Efficient serialization for large-scale training data with reduced I/O overhead | TFX, Dataflow, TensorFlow, Custom Training |
| **Data Fusion** | Visual ETL with 150+ connectors | Code-free multi-source data integration and transformation | BigQuery, Cloud Storage, Dataflow |
| **DataPrep** | Intelligent data preparation tool | Interactive no-code data cleaning and wrangling for analysts | BigQuery, AutoML, Cloud Storage |
| **DataProc** | Managed Apache Spark and Hadoop service | Spark MLlib training, large-scale PySpark ETL, migrate on-prem Hadoop workloads | Cloud Storage, BigQuery, Vertex AI, Spark MLlib |
| **Dataplex** | Unified data lake management and governance | Centralized metadata, data quality, security policies across data lakes and warehouses | BigQuery, Feature Store, Cloud Storage, Data Catalog |
| **Cloud Bigtable** | NoSQL wide-column database | Low-latency, high-throughput feature storage for online serving | Feature Store, Endpoints, Dataflow, HBase API |
| **Cloud Document AI** | Specialized document processing with OCR and parsing | Extract structured data from invoices, receipts, contracts, forms | Vision AI API, AutoML NLP, Natural Language AI, Cloud Storage |
| **Vision AI API** | Pre-trained vision models (OCR, label/face/object detection, SafeSearch) | Image analysis without custom training, content moderation | AutoML Vision, Document AI, Video Intelligence, Cloud Storage |
| **Natural Language AI API** | Pre-trained NLP (entity extraction, sentiment, syntax, content classification) | Text analysis without custom model training | AutoML NLP, Translation API, Document AI |
| **Translation API** | Neural machine translation (NMT) | Translate text and documents between 100+ languages | Natural Language AI, Document AI, AutoML NLP |
| **Speech-to-Text API** | Automatic speech recognition (ASR) | Convert audio to text with speaker diarization, support for 125+ languages | Natural Language AI, Dialogflow, Video Intelligence, Cloud Storage |
| **Text-to-Speech API** | Neural voice synthesis with WaveNet | Generate natural-sounding speech from text in multiple voices and languages | Dialogflow, Contact Center AI, Speech-to-Text |
| **Video Intelligence API** | Pre-trained video analysis (label/shot/object detection, transcription) | Analyze video content, detect scenes, extract metadata without custom training | Vision AI, Speech-to-Text, Cloud Storage |
| **Recommendations AI** | Managed ML-powered recommendation engine | E-commerce product recommendations, media personalization, similar item suggestions | BigQuery, Retail API, Matching Engine, Endpoints |
| **Cloud DLP API** | Sensitive data discovery, classification, and de-identification | Detect and redact PII before training, ensure data compliance (GDPR, HIPAA) | BigQuery, Cloud Storage, Dataflow, Pipelines |
| **Kubeflow Pipelines (KFP)** | Open-source pipeline engine | Multi-cloud or non-GCP pipelines | Vertex Pipelines, Kubernetes, TFX, custom ML workflows |
| **TensorFlow Extended (TFX)** | Full ML pipeline framework | TF-heavy production ML | Dataflow, TFRecord, Vertex Pipelines, TensorFlow, Beam |
| **TensorFlow Serving** | TF model server | Self-hosted TF inference | Vertex Endpoints, Kubernetes, Docker, TensorFlow models |
| **Cloud Build** | Continuous integration/continuous deployment | Build custom containers, automate ML pipeline deployment, CI/CD for models | Artifact Registry, Cloud Run, Pipelines, GitHub, GitLab |
| **Artifact Registry** | Universal package and container registry | Store Docker images, Python packages, model artifacts | Custom Training, Cloud Build, Deep Learning Containers, Cloud Run |
| **Container Registry (legacy)** | Docker container registry | Store container images (superseded by Artifact Registry) | Artifact Registry, GKE, Cloud Run |
| **Cloud Monitoring** | Infrastructure and application monitoring | Track model performance metrics, training job monitoring, resource utilization | Model Monitoring, Endpoints, Custom Training, Cloud Logging |
| **Cloud Logging** | Centralized logging service | Debug training jobs, track prediction requests, audit trails | Custom Training, Endpoints, Pipelines, Cloud Monitoring |
| **Model Garden** | Pretrained/foundation models | Using or fine-tuning SOTA models | Vertex Endpoints, Custom Training, Pipelines, Hugging Face integration |
| **Cloud Storage (GCS)** | Scalable object storage with lifecycle management | Store datasets, trained models, pipeline artifacts, staging data | All Vertex AI services, BigQuery, Dataflow, Pipelines |
| **Pub/Sub** | Real-time messaging and event streaming | Ingest streaming data, trigger ML pipelines, decouple microservices | Dataflow, Cloud Functions, Cloud Run, Pipelines |
| **Cloud Run** | Fully managed serverless containers | Deploy custom prediction servers, lightweight model serving, event-driven ML pipelines | Cloud Build, Artifact Registry, Pub/Sub, Cloud Storage |
| **Cloud Functions** | Event-driven serverless functions | Trigger preprocessing jobs, lightweight data transformations, pipeline orchestration | Pub/Sub, Cloud Storage, Cloud Scheduler, Pipelines |
| **Cloud Composer (Airflow)** | Fully managed Apache Airflow for workflow orchestration | Complex DAG scheduling, hybrid data/ML workflows, cross-service orchestration | Vertex Pipelines, BigQuery, Dataflow, Dataproc, Cloud Storage |



## GCP ML Services vs AWS Services

| GCP Service | AWS Equivalent | Primary Use Case |
|-------------|----------------|------------------|
| Vertex AI Pipelines | SageMaker Pipelines | ML workflow orchestration |
| Vertex AI Custom Training | SageMaker Training Jobs | Custom model training |
| Vertex AI Batch Prediction | SageMaker Batch Transform | Batch inference |
| Vertex AI Notebooks | SageMaker Studio Notebooks | Interactive development |
| Vertex AI Workbench | SageMaker Studio | Managed ML IDE |
| Vertex AI Notebook Runner | SageMaker Studio Jobs | Scheduled notebook execution |
| Vertex AI AutoML | SageMaker Autopilot, Canvas | Automated ML |
| Vertex AI Experiments | SageMaker Experiments | Experiment tracking |
| Vertex AI Metadata | SageMaker ML Lineage Tracking | Artifact lineage |
| Vertex AI TensorBoard | SageMaker Debugger, TensorBoard on SageMaker | Training visualization |
| Vertex AI Managed Dataset | SageMaker Data Wrangler | Dataset management |
| Vertex AI Feature Store | SageMaker Feature Store | Feature management |
| Vertex AI Model Registry | SageMaker Model Registry | Model versioning |
| Vertex AI Endpoint | SageMaker Real-time Endpoints | Real-time inference |
| Vertex AI Model Monitoring | SageMaker Model Monitor | Drift detection |
| Vertex Explainable AI | SageMaker Clarify | Model interpretability |
| Vertex AI Hyperparameter Tuning | SageMaker Hyperparameter Tuning | HPO |
| Vertex AI Vizier | SageMaker Automatic Model Tuning | Black-box optimization |
| Vertex AI Matching Engine | OpenSearch Service, Kendra | Vector similarity search |
| Vertex AI Labeling | SageMaker Ground Truth | Data labeling |
| Cloud TPU | AWS Trainium, Inferentia | ML accelerators |
| Deep Learning VM Images | EC2 Deep Learning AMI | Pre-configured ML VMs |
| Deep Learning Containers | AWS Deep Learning Containers | ML framework containers |
| BigQuery | Amazon Redshift, Athena | Data warehouse |
| BigQuery ML | Redshift ML, Athena ML | SQL-based ML |
| BigQuery Scheduled Queries | EventBridge + Athena/Redshift | Scheduled queries |
| Dataflow | AWS Glue, Kinesis Data Analytics | Stream/batch processing |
| Data Fusion | AWS Glue Studio | Visual ETL |
| DataPrep | AWS Glue DataBrew | Data preparation |
| Dataproc | Amazon EMR | Spark/Hadoop workloads |
| Dataplex | AWS Lake Formation | Data lake governance |
| Cloud Bigtable | DynamoDB, Keyspaces (Cassandra) | NoSQL wide-column store |
| Cloud Document AI | Amazon Textract | Document extraction |
| Vision AI API | Amazon Rekognition | Image analysis |
| Natural Language AI API | Amazon Comprehend | Text analysis |
| Translation API | Amazon Translate | Language translation |
| Speech-to-Text API | Amazon Transcribe | Speech recognition |
| Text-to-Speech API | Amazon Polly | Speech synthesis |
| Video Intelligence API | Amazon Rekognition Video | Video analysis |
| Recommendations AI | Amazon Personalize | Recommendation systems |
| Cloud DLP API | Amazon Macie | Data loss prevention |
| Kubeflow Pipelines (KFP) | Kubeflow on EKS | Open-source ML pipelines |
| TensorFlow Extended (TFX) | SageMaker Pipelines (TFX components) | TensorFlow ML pipelines |
| TensorFlow Serving | TorchServe, TensorFlow Serving on EC2 | Model serving |
| Model Garden | SageMaker JumpStart | Pre-trained models |
| Cloud Build | AWS CodeBuild | CI/CD |
| Artifact Registry | Amazon ECR, CodeArtifact | Artifact storage |
| Container Registry | Amazon ECR | Container registry |
| Cloud Monitoring | Amazon CloudWatch | Monitoring |
| Cloud Logging | Amazon CloudWatch Logs | Logging |
| Cloud Storage (GCS) | Amazon S3 | Object storage |
| Pub/Sub | Amazon Kinesis, SNS/SQS | Messaging/streaming |
| Cloud Run | AWS Fargate, App Runner | Serverless containers |
| Cloud Functions | AWS Lambda | Serverless functions |
| Cloud Composer | Amazon MWAA (Managed Airflow) | Workflow orchestration |

## When to Use Which Service

| Use Case | GCP Service | Why |
|----------|-------------|-----|
| Quick PoC with tabular data | BigQuery ML or AutoML Tabular | No ML expertise needed, SQL-based or automated |
| Custom deep learning (TensorFlow/PyTorch) | Vertex AI Custom Training | Full control, custom code, TPU/GPU access |
| Large-scale batch predictions | Vertex AI Batch Prediction | Cost-effective, BigQuery integration, async processing |
| Real-time predictions with low latency | Vertex AI Endpoints | Auto-scaling, managed infrastructure, traffic splitting |
| Streaming data processing and transformations | Dataflow | Unified batch/streaming, Apache Beam, exactly-once processing |
| Visual ETL for ML data prep | Cloud Data Fusion | No-code, 150+ connectors, business user friendly |
| Code-based data preprocessing at scale | Dataflow with TensorFlow Transform | Training-serving consistency, distributed processing |
| Data warehouse analytics and SQL | BigQuery | Serverless, petabyte-scale, fast analytics |
| End-to-end ML pipeline orchestration | Vertex AI Pipelines | ML-specific, metadata tracking, reproducibility |
| Automated hyperparameter optimization | Vertex AI Hyperparameter Tuning or Vizier | Bayesian optimization, distributed trials |
| Time series forecasting | BigQuery ML ARIMA_PLUS or AutoML Forecasting | SQL-based or automated, seasonality detection |
| Image classification without custom training | Vision AI API | Pre-trained models, immediate deployment |
| Custom image classification | Vertex AI AutoML Vision | Custom classes, transfer learning, no code |
| Advanced custom vision models | Vertex AI Custom Training + TensorFlow/PyTorch | Full control, custom architectures, fine-tuning |
| Text analysis and NLP (pre-trained) | Natural Language AI API | Entity extraction, sentiment, syntax analysis |
| Custom text classification | Vertex AI AutoML NLP | Domain-specific text models, no code required |
| Document parsing and data extraction | Cloud Document AI | Invoices, receipts, forms, OCR + structure |
| Video content analysis | Video Intelligence API | Shot detection, labels, explicit content detection |
| Recommendation systems (e-commerce) | Recommendations AI | Managed recommendations, retail-optimized |
| Vector similarity search at scale | Vertex AI Matching Engine | Semantic search, low-latency embeddings matching |
| Real-time feature serving | Vertex AI Feature Store + Bigtable | Low-latency online serving, point-in-time correctness |
| Offline feature computation | Vertex AI Feature Store + BigQuery | Batch feature generation, historical features |
| Feature engineering for training/serving consistency | Dataflow + TensorFlow Transform | Same preprocessing in training and serving |
| Model versioning and lineage tracking | Vertex AI Model Registry | Manage versions, track lineage, promote models |
| Production model monitoring | Vertex AI Model Monitoring | Drift detection, skew detection, alerting |
| Model explainability and interpretability | Vertex Explainable AI | Feature attributions, Shapley values, compliance |
| Experiment tracking and comparison | Vertex AI Experiments + TensorBoard | Compare runs, visualize metrics, hyperparameters |
| Data labeling with human annotators | Vertex AI Labeling | Managed workforce, quality control |
| Training large models with TPUs | Cloud TPU + Vertex AI Custom Training | Accelerated training, TensorFlow/JAX/PyTorch |
| Quick ML development environment | Deep Learning VM Images or Vertex AI Workbench | Pre-configured frameworks, GPU/TPU support |
| Containerized custom training | Deep Learning Containers + Vertex AI Training | Reproducible environments, custom dependencies |
| Spark-based ML and ETL | Dataproc with Spark MLlib | Existing Spark workflows, large-scale ETL |
| Complex workflow orchestration (ETL + ML) | Cloud Composer (Airflow) | Hybrid workflows, scheduling, cross-service |
| Event-driven ML workflows | Cloud Functions or Pub/Sub + Dataflow | Trigger pipelines on events, real-time responses |
| Lightweight custom model serving | Cloud Run with custom container | Serverless, cost-effective for low traffic |
| CI/CD for ML models | Cloud Build + Artifact Registry + Vertex Pipelines | Automated testing, container builds, deployment |
| Sensitive data handling and PII detection | Cloud DLP API | Detect/redact PII, compliance (GDPR, HIPAA) |
| SQL-native machine learning | BigQuery ML | In-warehouse ML, no data movement, SQL skills |
| Multi-language translation | Translation API | 100+ languages, neural translation |
| Speech-to-text transcription | Speech-to-Text API | ASR, speaker diarization, 125+ languages |
| Text-to-speech synthesis | Text-to-Speech API | Natural voices, WaveNet, multiple languages |
| No-code ML for business users | Vertex AI AutoML or BigQuery ML | Point-and-click or SQL-based, no coding |
| Production-grade TensorFlow pipelines | TFX on Dataflow + Vertex Pipelines | End-to-end TF workflows, validation, serving |
| Data lake governance and cataloging | Dataplex | Metadata management, data quality, security zones |
| Interactive data cleaning | DataPrep | Visual data wrangling, data profiling |
| Pre-trained foundation models | Model Garden | SOTA models, fine-tuning, quick deployment |