# Simulation-Driven ML for Semi-Dynamic Project Delay Prediction (DS 340W Capstone)

A simulation-driven machine learning framework that predicts project schedule delay early by combining:
- **CPM/PERT** project structure,
- **Monte Carlo simulation–derived uncertainty features**, and
- early **EVM indicators (SPI/CPI at 20% progress)**.

We train classical ML models to learn delay-risk patterns **from simulation outcomes** rather than relying on expert-defined fuzzy/grey rules or predefined causal structures.

**Team:** Sahana Ramachandran, Varsha Giridharan, Siyona Behera
**Course:** DS 340W (Fall 2025)

---

## Why this matters

Many traditional forecasting methods (CPM/PERT, fuzzy EVM, classical MC) are effectively **static snapshots** and often treat risks as independent. This project uses simulation as a **data generator** to extract empirical uncertainty behavior and then trains ML models for **semi-dynamic early forecasting** (20% progress).

---

## Method (high level)

1. **Data + structure**
   - Uses RanGen2 **RG30** synthetic project networks (Set 1; 900 projects).

2. **Uncertainty injection (PERT triplets)**
   - Most-likely duration comes from RG30.
   - Optimistic/pessimistic durations are generated around it (controlled spread).

3. **Monte Carlo simulation**
   - **200 runs per project**
   - Produces distributional / uncertainty features (e.g., instability from variance of completion times).
   - A project is labeled **delayed** if the probability of late completion exceeds a threshold (see report).

4. **Early progress features (20% progress)**
   - Computes **SPI** and **CPI** at 20% progress using time-based proxies (RG30 has no cost). 
   
5. **Models**
   - Logistic Regression
   - Gaussian Naive Bayes
   - Random Forest (primary non-linear model)

6. **Baselines**
   - Early SPI thresholding
   - Structural-only logistic regression baseline

---

## Key results (summary)

- **Random Forest** performed best overall and outperformed early-SPI baseline.
- Structural CPM features are strong; adding simulation-derived uncertainty + early CPI improves predictive signal.
- Reported test performance is around **ROC-AUC ≈ 0.76** for the tuned Random Forest (see report tables/figures). 

---

## Repository contents (suggested clean structure)

capstone-delay-prediction/
├─ README.md
├─ report/
│ └─ Report.pdf
├─ notebooks/
│ └─ FinalCode.ipynb
├─ data/
│ └─ rg30_set1.csv
└─ assets/
└─ (optional) figures, screenshots


---

## How to run (Colab)

1. Download:
   - `notebooks/FinalCode.ipynb`
   - `data/rg30_set1.csv`

2. Upload to Google Drive (MyDrive).

3. Open the notebook in Google Colab and **Run all**.

> Note: You do **not** need to run the full data-generation section to reproduce results (final dataset is provided).

---

## Local setup (optional)

If you want to run locally:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Then open the notebook in Jupyter/VSCode.

Data source
Project network data is based on Ghent University Project Management Research Group datasets (RanGen/RG30). 
Report


Citation
If you use or build on this work, please cite the course report in report/Report.pdf.


