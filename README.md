# Simulation-Driven ML for Semi-Dynamic Project Delay Prediction  
*(DS 340W Capstone + Engineering Extension)*

This project explores **early prediction of project schedule delays** by combining:

- **CPM/PERT project structure**
- **Monte Carlo simulation–derived uncertainty features**
- **Early earned-value indicators (SPI/CPI at 20% progress)**

In addition to the original research component, the project has been extended into a reproducible ML pipeline with explicit training, inference, experiment tracking, and containerization.

---

## Research motivation

Many traditional forecasting approaches (CPM/PERT, fuzzy EVM, classical Monte Carlo) operate as **static snapshots** and often assume independent risks. This work uses **simulation as a data generator** to capture empirical uncertainty behavior, then trains ML models to enable **semi-dynamic early forecasting** at only 20% project progress.

---

## Method (high level)

### Data + structure
- Synthetic project networks from **RanGen2 RG30** (Set 1; 900 projects)

### Uncertainty injection (PERT triplets)
- Most-likely durations from RG30
- Optimistic/pessimistic durations generated with controlled spread

### Monte Carlo simulation
- 200 simulation runs per project
- Distributional uncertainty features (e.g., instability from variance of completion times)
- Project labeled *delayed* if probability of late completion exceeds a threshold

### Early progress features (20%)
- SPI and CPI computed at 20% progress
- Time-based cost proxies (RG30 has no explicit cost data)

### Models
- Logistic Regression  
- Gaussian Naive Bayes  
- **Random Forest (primary non-linear model)**

### Baselines
- Early-SPI thresholding  
- Structural-only logistic regression  

---

## Key results (summary)

- Random Forest performs best overall and outperforms early-SPI baseline  
- Structural CPM features are strong predictors  
- Adding simulation-derived uncertainty and early CPI improves signal  
- Test performance: **ROC-AUC ≈ 0.76** for tuned Random Forest  
  (see tables/figures in report)

---

## Engineering extension

The notebook-based research code has been refactored into a **reproducible ML system**:

### Training
- Deterministic data splits  
- Explicit hyperparameter configuration  
- Model evaluation (Accuracy, F1, ROC-AUC)  
- Model artifact persistence  
- MLflow experiment tracking (params, metrics, artifacts, model)

### Inference
- Schema-validated JSON input  
- Shared preprocessing logic (no training–inference drift)  
- Single-instance probability prediction:  P(delay | project features)
- CLI interface

### Reproducibility
- Dockerized training and inference  
- Consistent runtime environment across machines  
- Artifacts written back via volume mounts  

---

## Repository structure
```
project-delay-prediction/
├─ src/
│ ├─ train.py # training + evaluation + MLflow logging
│ ├─ infer.py # inference CLI
│ ├─ data.py # data loading + preprocessing
│ ├─ schema.py # feature / label contract
│
├─ data/
│ └─ rg30_set1.csv
│
├─ models/
│ └─ rf_delay.joblib
│
├─ report/
│ └─ Report.pdf
│
├─ examples/
│ └─ ex1.json # example inference input
│
├─ Dockerfile
├─ requirements.txt
└─ README.md
```

---

## How to run (Docker)

### Inference
```
docker run --rm -v "$PWD:/app" delay-predictor \
  python -m src.infer --input examples/ex1.json
```
### Training
```
docker run --rm -v "$PWD:/app" delay-predictor \
  python -m src.train
```

- Artifacts (models/, report/, mlruns/) are written to the local filesystem.

## How to run (local)

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m src.train
```

## Original research context

This project was developed as part of the DS 340W capstone at Penn State.
See `report/Report.pdf` for full methodological details and results.

- Team: Sahana Ramachandran, Varsha Giridharan, Siyona Behera
- Data source: Ghent University Project Management Research Group (RanGen / RG30)


