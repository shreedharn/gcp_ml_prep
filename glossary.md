# GCP ML Terminology Glossary

A comprehensive reference guide to key terms, services, and acronyms used in Google Cloud Machine Learning, with AWS equivalents where applicable.

## Table of Contents

- [A](#a) | [B](#b) | [C](#c) | [D](#d) | [E](#e) | [F](#f) | [G](#g) | [H](#h) | [I](#i) | [K](#k) | [L](#l) | [M](#m) | [N](#n) | [O](#o) | [P](#p) | [R](#r) | [S](#s) | [T](#t) | [U](#u) | [V](#v) | [W](#w)

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

### Convergence

| Attribute | Details |
|-----------|---------|
| **Description** | The point during training where model performance stops improving significantly |
| **Use Case** | Determining when to stop training to avoid wasted compute resources |

---

## D

### DAG (Directed Acyclic Graph)

| Attribute | Details |
|-----------|---------|
| **Description** | Graph structure representing ML pipeline dependencies |
| **Use Case** | Workflow orchestration in Vertex AI Pipelines, Kubeflow |

### Cloud Data Fusion

| Attribute | Details |
|-----------|---------|
| **Description** | Fully managed, cloud-native data integration service with visual ETL/ELT pipeline builder |
| **AWS Equivalent** | AWS Glue Studio, AWS Glue DataBrew |
| **Key Features** | Visual drag-and-drop, 150+ connectors, data quality checks, pipeline templates |
| **Use Case** | Data preparation for ML, consolidating multiple sources, business-user friendly ETL |

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

## E

### Epoch

| Attribute | Details |
|-----------|---------|
| **Description** | One complete pass through the entire training dataset |
| **Use Case** | Measuring training progress and controlling training duration |

---

## F

### False Negative (FN)

| Attribute | Details |
|-----------|---------|
| **Description** | Actual positive case incorrectly predicted as negative |
| **Use Case** | Critical in medical diagnosis and fraud detection where missing positives is costly |

### False Positive (FP)

| Attribute | Details |
|-----------|---------|
| **Description** | Actual negative case incorrectly predicted as positive |
| **Use Case** | Important in spam detection where false alarms reduce user trust |

### Feature

| Attribute | Details |
|-----------|---------|
| **Description** | An individual measurable property or characteristic used as input to a model |
| **Use Case** | Building predictive models from structured data |

### Feature Store

| Attribute | Details |
|-----------|---------|
| **Description** | Centralized repository for storing, serving, and managing ML features |
| **AWS Equivalent** | SageMaker Feature Store |
| **Components** | Online store (real-time), Offline store (training) |

---

## G

### Generalization

| Attribute | Details |
|-----------|---------|
| **Description** | A model's ability to perform well on new, unseen data |
| **Use Case** | Core objective of machine learning; preventing overfitting |

### Gradient Descent

| Attribute | Details |
|-----------|---------|
| **Description** | Optimization algorithm that iteratively adjusts weights to minimize loss |
| **Use Case** | Training neural networks and other ML models |

---

## H

### Hyperparameter

| Attribute | Details |
|-----------|---------|
| **Description** | Configuration value set before training that controls the learning process (e.g., learning rate) |
| **Use Case** | Tuning model performance through systematic optimization |

---

## I

### IAM (Identity and Access Management)

| Attribute | Details |
|-----------|---------|
| **Description** | Access control service for GCP resources |
| **AWS Equivalent** | AWS IAM |
| **Key Roles** | aiplatform.user, aiplatform.admin, bigquery.dataViewer |

### Iteration

| Attribute | Details |
|-----------|---------|
| **Description** | One update of model weights, typically after processing one batch |
| **Use Case** | Tracking training progress within an epoch |

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

## L

### Lambda (λ)

| Attribute | Details |
|-----------|---------|
| **Description** | Regularization strength parameter; higher values increase penalty on model complexity |
| **Use Case** | Controlling the tradeoff between fitting training data and model simplicity |

### Loss Function

| Attribute | Details |
|-----------|---------|
| **Description** | Mathematical function that quantifies the difference between predictions and actual values |
| **Use Case** | Guiding model optimization during training |

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

## N

### Noise

| Attribute | Details |
|-----------|---------|
| **Description** | Random variations or errors in data that don't represent true patterns |
| **Use Case** | Understanding data quality and the limits of model accuracy |

---

## O

### Overfitting

| Attribute | Details |
|-----------|---------|
| **Description** | Model performs well on training data but poorly on new data due to memorization |
| **Use Case** | Identifying when model complexity needs to be reduced or regularization applied |

---

## P

### Parameter

| Attribute | Details |
|-----------|---------|
| **Description** | Internal variable learned by the model from data (e.g., weights, biases) |
| **Use Case** | Understanding model capacity and complexity |

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

### Sparsity

| Attribute | Details |
|-----------|---------|
| **Description** | Having many zero values; L1 regularization creates sparse weight matrices |
| **Use Case** | Feature selection and model interpretability |

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

### True Negative (TN)

| Attribute | Details |
|-----------|---------|
| **Description** | Actual negative case correctly predicted as negative |
| **Use Case** | Calculating accuracy, specificity, and other classification metrics |

### True Positive (TP)

| Attribute | Details |
|-----------|---------|
| **Description** | Actual positive case correctly predicted as positive |
| **Use Case** | Calculating precision, recall, and other classification metrics |

---

## U

### Underfitting

| Attribute | Details |
|-----------|---------|
| **Description** | Model is too simple to capture data patterns, performing poorly on all data |
| **Use Case** | Identifying when model complexity needs to be increased |

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

## W

### Weight

| Attribute | Details |
|-----------|---------|
| **Description** | Learned parameter that determines the importance of a feature in making predictions |
| **Use Case** | Understanding model decisions and feature importance |

---

This glossary provides essential terminology for Google Cloud ML certification preparation, with AWS comparisons to leverage existing cloud knowledge.
