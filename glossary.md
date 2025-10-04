# GCP ML Terminology Glossary

A comprehensive reference guide to key terms, services, and acronyms used in Google Cloud Machine Learning, with AWS equivalents where applicable.

## Table of Contents

- [A](#a) | [B](#b) | [C](#c) | [D](#d) | [F](#f) | [I](#i) | [K](#k) | [M](#m) | [P](#p) | [R](#r) | [S](#s) | [T](#t) | [V](#v)

---

## A

### AutoML

| Attribute | Details |
|-----------|---------|
| **Description** | Automated machine learning service that automates model selection, hyperparameter tuning, and deployment |
| **AWS Equivalent** | SageMaker Autopilot |
| **Use Case** | Quick ML development without deep ML expertise |

### AUC (Area Under Curve)

| Attribute | Details |
|-----------|---------|
| **Description** | Performance metric measuring the area under the ROC curve |
| **Range** | 0.0 to 1.0 (higher is better) |
| **Use Case** | Evaluating binary classification models |

---

## B

### BigQuery ML

| Attribute | Details |
|-----------|---------|
| **Description** | Create and execute ML models using SQL directly in BigQuery |
| **AWS Equivalent** | Redshift ML |
| **Key Features** | SQL-based model training, no data movement required |

---

## C

### CMEK (Customer-Managed Encryption Keys)

| Attribute | Details |
|-----------|---------|
| **Description** | User-controlled encryption keys for data security |
| **AWS Equivalent** | Customer managed keys in KMS |
| **Use Case** | Regulatory compliance, data sovereignty |

### CT (Continuous Training)

| Attribute | Details |
|-----------|---------|
| **Description** | Automated retraining of models based on triggers or schedules |
| **Components** | Data monitoring, training pipeline, deployment automation |

---

## D

### DAG (Directed Acyclic Graph)

| Attribute | Details |
|-----------|---------|
| **Description** | Graph structure representing ML pipeline dependencies |
| **Use Case** | Workflow orchestration in Vertex AI Pipelines, Kubeflow |

### Dataflow

| Attribute | Details |
|-----------|---------|
| **Description** | Managed service for stream and batch data processing based on Apache Beam |
| **AWS Equivalent** | AWS Glue (batch), Kinesis Analytics (stream) |
| **Key Features** | Unified programming model, auto-scaling, windowing |

### DNN (Deep Neural Network)

| Attribute | Details |
|-----------|---------|
| **Description** | Neural network with multiple hidden layers |
| **Use Case** | Complex pattern recognition, computer vision, NLP |

---

## F

### Feature Store

| Attribute | Details |
|-----------|---------|
| **Description** | Centralized repository for storing, serving, and managing ML features |
| **AWS Equivalent** | SageMaker Feature Store |
| **Components** | Online store (real-time), Offline store (training) |

---

## I

### IAM (Identity and Access Management)

| Attribute | Details |
|-----------|---------|
| **Description** | Access control service for GCP resources |
| **AWS Equivalent** | AWS IAM |
| **Key Roles** | aiplatform.user, aiplatform.admin, bigquery.dataViewer |

---

## K

### KFP (Kubeflow Pipelines)

| Attribute | Details |
|-----------|---------|
| **Description** | ML workflow framework used in Vertex AI Pipelines |
| **AWS Equivalent** | SageMaker Pipelines (different framework) |
| **Components** | Pipeline definition, components, artifacts |

### KMS (Key Management Service)

| Attribute | Details |
|-----------|---------|
| **Description** | Service for creating and managing cryptographic keys |
| **AWS Equivalent** | AWS KMS |
| **Use Case** | Encryption key management, CMEK implementation |

---

## M

### MAE (Mean Absolute Error)

| Attribute | Details |
|-----------|---------|
| **Description** | Average of absolute differences between predictions and actual values |
| **Formula** | MAE = (1/n) × Σ\|predicted - actual\| |
| **Use Case** | Regression model evaluation |

### MSE (Mean Squared Error)

| Attribute | Details |
|-----------|---------|
| **Description** | Average of squared differences between predictions and actual values |
| **Formula** | MSE = (1/n) × Σ(predicted - actual)² |
| **Use Case** | Regression model evaluation, sensitive to outliers |

### MWAA (Managed Workflows for Apache Airflow)

| Attribute | Details |
|-----------|---------|
| **Description** | AWS managed Apache Airflow service |
| **GCP Equivalent** | Cloud Composer |
| **Use Case** | Workflow orchestration, ETL pipelines |

---

## P

### Pub/Sub

| Attribute | Details |
|-----------|---------|
| **Description** | Messaging service for event ingestion and distribution |
| **AWS Equivalent** | Amazon Kinesis, SNS/SQS |
| **Key Features** | At-least-once delivery, ordering, topic-based routing |

---

## R

### RMSE (Root Mean Squared Error)

| Attribute | Details |
|-----------|---------|
| **Description** | Square root of MSE, in same units as target variable |
| **Formula** | RMSE = √MSE |
| **Use Case** | Regression model evaluation |

### ROC (Receiver Operating Characteristic)

| Attribute | Details |
|-----------|---------|
| **Description** | Plot of true positive rate vs false positive rate |
| **Use Case** | Visualizing classification performance at different thresholds |

---

## S

### SHAP (SHapley Additive exPlanations)

| Attribute | Details |
|-----------|---------|
| **Description** | Method for explaining individual predictions |
| **Use Case** | Model interpretability, feature importance |

### SMOTE (Synthetic Minority Over-sampling Technique)

| Attribute | Details |
|-----------|---------|
| **Description** | Technique for handling imbalanced datasets by generating synthetic examples |
| **Use Case** | Improving minority class representation |

---

## T

### Tabular Workflows

| Attribute | Details |
|-----------|---------|
| **Description** | Enhanced AutoML with granular pipeline control |
| **AWS Equivalent** | Custom SageMaker Pipelines |
| **Use Case** | Structured data ML with customization needs |

### TFT (TensorFlow Transform)

| Attribute | Details |
|-----------|---------|
| **Description** | Library for preprocessing data in TensorFlow |
| **Key Feature** | Ensures training-serving consistency |
| **Use Case** | Feature engineering, preprocessing pipelines |

### TFDV (TensorFlow Data Validation)

| Attribute | Details |
|-----------|---------|
| **Description** | Library for exploring and validating ML data |
| **Capabilities** | Schema inference, anomaly detection, drift detection |

### TFX (TensorFlow Extended)

| Attribute | Details |
|-----------|---------|
| **Description** | End-to-end platform for deploying production ML pipelines |
| **Components** | Data validation, transform, training, serving |

### TPU (Tensor Processing Unit)

| Attribute | Details |
|-----------|---------|
| **Description** | Google's custom chip optimized for ML workloads (GCP exclusive) |
| **AWS Equivalent** | AWS Trainium/Inferentia (similar concept) |
| **Best For** | Large-scale training, matrix operations, TensorFlow models |

---

## V

### Vertex AI

| Attribute | Details |
|-----------|---------|
| **Description** | Unified ML platform consolidating training, deployment, and monitoring |
| **AWS Equivalent** | Amazon SageMaker |
| **Key Components** | Workbench, Training, Prediction, Pipelines, Feature Store |

### VPC (Virtual Private Cloud)

| Attribute | Details |
|-----------|---------|
| **Description** | Isolated network environment for GCP resources |
| **AWS Equivalent** | Amazon VPC |
| **Use Case** | Network security, resource isolation |

### VPC-SC (VPC Service Controls)

| Attribute | Details |
|-----------|---------|
| **Description** | Security perimeter for GCP resources to prevent data exfiltration |
| **AWS Equivalent** | VPC Endpoints, PrivateLink |
| **Use Case** | Data security, compliance requirements |

---

This glossary provides essential terminology for Google Cloud ML certification preparation, with AWS comparisons to leverage existing cloud knowledge.
