# HeveMind
### An Agentic Explainable AI Decision Support Platform for Semiconductor Yield Optimization

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Production-green)
![Explainable AI](https://img.shields.io/badge/XAI-SHAP-orange)
![Status](https://img.shields.io/badge/Project-Completed-success)


# Project Ownership

HeveMind is an independent portfolio and research project developed entirely by **Barkavi A/P P Cheven**.

The following components were independently designed and implemented by the author:

- Overall system architecture
- Data engineering pipeline
- Feature engineering framework
- Machine learning workflow
- Explainable AI (SHAP)
- Evidence fusion engine
- Historical similarity engine
- Agentic AI orchestration
- Model monitoring and drift detection
- Deployment framework
- FastAPI backend
- Streamlit dashboards
- Governance and security modules
- Technical documentation
- Architecture diagrams

Third-party open-source libraries such as Scikit-learn, XGBoost, SHAP, FastAPI, and Streamlit are used as implementation tools. The overall system design, software engineering, integration, and project architecture are the original work of the author.
---

## Overview

HeveMind is an end-to-end industrial Artificial Intelligence platform designed to support semiconductor manufacturing through intelligent wafer risk prediction, explainable decision support, operational monitoring, uncertainty-aware inference, and deployment governance.

Unlike traditional machine learning projects that terminate after model training, HeveMind simulates a production-ready AI system by integrating:

- Machine Learning
- Explainable AI (XAI)
- Decision Intelligence
- Model Governance
- Drift Detection
- Operational Monitoring
- Human-in-the-loop Engineering Support

The project demonstrates how modern AI systems can be deployed responsibly in high-value manufacturing environments where prediction reliability, interpretability, and operational risk are equally important.

---

# Project Objectives

The primary objectives of HeveMind are to:

- predict semiconductor wafer failures before fabrication completion
- reduce unnecessary engineering investigations
- provide transparent AI-assisted decisions
- quantify prediction uncertainty
- recommend engineering actions
- monitor production model health
- support responsible AI deployment

---

# Industrial Motivation

In semiconductor manufacturing:

- defective wafers are extremely costly
- missed failures can propagate through downstream fabrication stages
- excessive false alarms increase engineering workload
- black-box AI systems are difficult to trust

HeveMind addresses these challenges by combining predictive analytics with explainable and uncertainty-aware AI.

---

# Dataset

Dataset:

SECOM Manufacturing Dataset

Characteristics:

- 1,567 wafers
- 590 sensor variables
- Highly imbalanced failure distribution
- Missing sensor measurements
- High-dimensional process variables

The pipeline automatically performs:

- data auditing
- missing value handling
- feature quality analysis
- anomaly detection
- feature engineering
- operational validation

---

# System Architecture

```
                 Manufacturing Sensors
                          │
                          ▼
               Data Quality Engine
                          │
                          ▼
          Missingness-Aware Processing
                          │
                          ▼
           Feature Engineering Pipeline
                          │
                          ▼
             Machine Learning Models
                          │
                          ▼
             Probability Calibration
                          │
                          ▼
          Three-Level Decision Engine
                          │
                          ▼
          Uncertainty Quantification
                          │
                          ▼
         Explainability & SHAP Engine
                          │
                          ▼
          Historical Evidence Fusion
                          │
                          ▼
      Engineering Decision Dashboard
                          │
                          ▼
      Drift Detection & Monitoring
```

---

# Key Features

## Data Engineering

- Automated data audit
- Missing value analysis
- Missingness-aware learning
- Constant feature detection
- Correlation analysis
- Feature validation
- Outlier detection

---

## Machine Learning

Implemented models include:

- Logistic Regression
- Random Forest
- Balanced Random Forest
- Easy Ensemble
- LightGBM
- XGBoost

Operational model selection is performed automatically using engineering-oriented evaluation criteria.

---

## Explainable AI

HeveMind provides:

- SHAP explanations
- Global feature importance
- Local wafer explanations
- Sensor investigation prioritisation
- Decision justification
- Root-cause assistance

---

## Calibration

Implemented calibration methods:

- Beta Calibration
- Isotonic Regression
- Platt Scaling (Sigmoid)

Probability calibration improves reliability before deployment.

---

## Decision Intelligence

Instead of binary predictions, wafers are classified into:

- Low Risk
- Engineering Review
- High Risk

Each decision includes recommended engineering actions.

---

## Uncertainty Quantification

Prediction reliability is evaluated using:

- prediction confidence
- uncertainty scores
- confidence bands
- uncertainty bands
- probability intervals
- abstention mechanism
- human review routing

---

## Evidence Fusion

Predictions are strengthened through:

- historical similarity retrieval
- nearest-neighbour evidence
- engineering evidence aggregation
- evidence fusion meta-model

---

## Operational Monitoring

Production monitoring includes:

- feature drift
- prediction drift
- deployment health score
- alert engine
- monitoring history
- retraining recommendation
- governance reporting

---

## Deployment Components

The project includes:

- Streamlit dashboard
- Executive PDF reports
- Excel engineering reports
- Audit logging
- Governance dashboard
- Scheduler support
- Command-line interface
- Production model persistence

---

# Repository Structure

```
HeveMind/

├── artifacts/
├── config/
├── dashboard/
├── data/
├── logs/
├── models/
├── reports/
├── src/
├── requirements.txt
└── README.md
```

---

# Results

The final deployed system includes:

- Operational model selection
- Cross-fitted probability calibration
- Three-level decision policy
- Explainable AI
- Uncertainty-aware inference
- Evidence fusion
- Drift monitoring
- Deployment health scoring
- Governance framework

The project evaluates models using:

- ROC-AUC
- PR-AUC
- Brier Score
- Balanced Accuracy
- Recall
- Precision
- F1-score
- Operational Cost
- Calibration Error
- Confidence Analysis

---

# Engineering Workflow

```
Raw Sensor Data
        │
        ▼
Quality Validation
        │
        ▼
Missingness Processing
        │
        ▼
Feature Engineering
        │
        ▼
Prediction
        │
        ▼
Calibration
        │
        ▼
Decision Policy
        │
        ▼
Explainability
        │
        ▼
Evidence Fusion
        │
        ▼
Uncertainty Analysis
        │
        ▼
Engineering Recommendation
        │
        ▼
Monitoring
```

---

# Technology Stack

Programming

- Python

Machine Learning

- Scikit-learn
- XGBoost
- LightGBM
- Imbalanced-learn

Explainability

- SHAP

Data Processing

- NumPy
- Pandas

Visualisation

- Matplotlib
- Plotly

Dashboard

- Streamlit

Model Storage

- Joblib

Reporting

- ReportLab
- OpenPyXL

Database

- SQLite

---

# Future Enhancements

Potential future developments include:

- online learning
- streaming sensor ingestion
- federated learning
- graph neural networks
- digital twin integration
- predictive maintenance
- reinforcement learning
- MES integration
- OPC-UA connectivity
- edge deployment

---

# Intended Audience

This repository is intended for:

- Semiconductor AI Engineers
- Data Scientists
- Machine Learning Engineers
- Manufacturing Engineers
- Explainable AI Researchers
- Industrial AI Researchers
- Graduate Recruiters
- Technical Interviewers

---

# Author

**Barkavi A/P P Cheven**

Master of Data Science

Universiti Malaya

Research Interests

- Explainable Artificial Intelligence
- Industrial AI
- Semiconductor Analytics
- Machine Learning
- Predictive Manufacturing
- Decision Intelligence

---

## License

This project is protected under an **All Rights Reserved** license.

The source code, architecture, documentation, diagrams, dashboards,
and software implementation are the intellectual property of
**Barkavi A/P P Cheven**.

See the LICENSE file for details.