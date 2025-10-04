# Common Scenarios and Solutions

## Scenario 1: Real-Time Fraud Detection

**Question**: Design a system for real-time credit card fraud detection with <50ms latency.

**Answer**:
```
User Transaction → API Gateway → Cloud Run (preprocessing)
                                     ↓
                              Feature Store (online serving)
                                     ↓
                              Vertex AI Endpoint (model)
                                     ↓
                                   Decision
```

**Key Points:**

- Feature Store for low-latency feature retrieval
- Vertex AI Endpoint with GPU for fast inference
- Cloud Run for stateless API layer
- Consider caching for repeat requests

## Scenario 2: Large-Scale Batch Predictions

**Question**: Process 100M predictions daily, cost-effectively.

**Answer**:

Follow this cost-effective approach:

- Use Vertex AI Batch Prediction (not online endpoints)
- Input from BigQuery, output to BigQuery
- Schedule with Cloud Scheduler
- Use preemptible VMs for 60-80% cost savings
- Partition output by date in BigQuery

## Scenario 3: Model Retraining on Drift

**Question**: Automatically retrain model when drift detected.

**Answer**:
```
Vertex AI Model Monitoring (drift detection)
             ↓
        Pub/Sub alert
             ↓
    Cloud Function (trigger)
             ↓
  Vertex AI Pipeline (retraining)
             ↓
    Model Registry (new version)
             ↓
   Endpoint (canary deployment)
```

## Scenario 4: SQL Analyst Needs ML

**Question**: Data analyst team wants to build ML models using SQL.

**Answer**:

BigQuery ML is the ideal solution:

- BigQuery ML is the right choice
- No Python/ML expertise required
- Data stays in BigQuery (no movement)
- Supports common model types (regression, classification, time series)

## Scenario 5: Training-Serving Consistency

**Question**: Ensure preprocessing same for training and serving.

**Answer**:

Use TensorFlow Transform to guarantee consistency:

- Use TensorFlow Transform (TFT)
- Define preprocessing_fn once
- Use in both training pipeline and serving
- Save transform function with model
- Apply same transform at inference time