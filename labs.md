# Hands-On Lab Recommendations

## Lab 1: End-to-End AutoML Pipeline

**Objective**: Build complete pipeline from data to deployed model

**Steps:**

1. Upload dataset to Cloud Storage
2. Create Vertex AI Dataset
3. Train AutoML model
4. Evaluate model performance
5. Deploy to endpoint
6. Make online predictions
7. Run batch predictions

**AWS Comparison**: Build same pipeline in SageMaker

## Lab 2: Custom Training with TensorFlow

**Objective**: Train custom model on Vertex AI

**Steps:**

1. Write TensorFlow training script
2. Create Docker container
3. Push to Artifact Registry
4. Create Custom Training Job
5. Monitor training with Cloud Logging
6. Export model to Cloud Storage
7. Deploy and serve

**AWS Comparison**: SageMaker Training Job with BYOC

## Lab 3: Streaming ML Pipeline

**Objective**: Real-time feature engineering and prediction

**Steps:**

1. Set up Pub/Sub topic
2. Create Dataflow streaming pipeline
3. Write to Feature Store (online)
4. Call Vertex AI Endpoint for predictions
5. Write results to BigQuery
6. Monitor pipeline

**AWS Comparison**: Kinesis + Flink + SageMaker

## Lab 4: MLOps with Vertex AI Pipelines

**Objective**: Build CI/CD/CT pipeline

**Steps:**

1. Write KFP pipeline definition
2. Set up Cloud Build trigger
3. Configure automated testing
4. Deploy pipeline on code commit
5. Set up model monitoring
6. Create retraining trigger

**AWS Comparison**: SageMaker Pipelines with CodePipeline

## Final Preparation Checklist

### ✅ Services Understanding

- [ ] Know all Vertex AI components
- [ ] Understand BigQuery ML capabilities
- [ ] Know when to use AutoML vs custom training
- [ ] Understand data processing services (Dataflow, Dataproc)

### ✅ Architectural Patterns

- [ ] Real-time prediction architecture
- [ ] Batch prediction pipeline
- [ ] Streaming ML pipeline
- [ ] MLOps/CI/CD patterns

### ✅ Technical Deep-Dives

- [ ] Hyperparameter tuning strategies
- [ ] Distributed training (data vs model parallelism)
- [ ] Feature Store (online vs offline serving)
- [ ] Model monitoring (skew vs drift)

### ✅ Best Practices

- [ ] Training-serving consistency (TFT)
- [ ] Security (CMEK, VPC-SC, IAM)
- [ ] Cost optimization (preemptible VMs, batch vs online)
- [ ] Performance optimization (TPUs, caching)

### ✅ AWS Comparisons

- [ ] Know equivalent services for all major GCP services
- [ ] Understand architectural differences
- [ ] Know unique GCP features (TPUs, BigQuery ML, Reduction Server)

**Good luck with your Google Cloud Professional Machine Learning Engineer certification!**