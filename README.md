# Predicting Data Drift in Production ML Models
## A Proactive Statistical Framework Using Distributional Divergence Metrics

## What Is This Project About?
Most machine learning projects end at test accuracy. This one begins there.
This repository presents a complete, end-to-end pipeline addressing a critical but often ignored question:

## What happens when your trained model encounters data that no longer resembles its training distribution?

This phenomenon is referred to as **silent model decay** , a gradual and invisible degradation in performance due to distribution shift in incoming data.

In cybersecurity systems, this is especially critical. A model trained on benign Monday traffic may silently fail when exposed to a Friday DDoS attack, without raising any alerts.

### Proposed Approach

This project introduces a proactive solution:

- Compute statistical divergence metrics:
  - Kullback-Leibler (KL) Divergence  
  - Jensen-Shannon (JS) Divergence  
  - Wasserstein Distance  
- Use these metrics as inputs to a regression-based meta-model  
- Predict model accuracy degradation before inference

The pipeline is built using the **CICIDS 2017 dataset**, containing over 2.8 million network flow records.

---

## The Central Result — In Plain Terms

| Scenario | KL Divergence | Model Accuracy |
|--------|-------------|---------------|
| Monday → Monday | ~0.00 | 100.00% |
| Monday → Tuesday | 0.71 | 96.90% |
| Monday → Wednesday | 7.88 | 63.50% |
| Monday → Thursday AM | 0.31 | 98.80% |
| Monday → Thursday PM | ~0.00 | 100.00% |
| Monday → Friday AM | 0.16 | 99.00% |
| Monday → Friday Port Scan | 12.14 | 44.50% |
| Monday → Friday DDoS | 12.37 | 43.33% |

### Key Insight
As divergence increases, model accuracy decreases — consistently and predictably.

### Meta-Model Performance

- **R²:** 0.9924  
- **MAE:** 1.43%  
- **Max Error:** ~0.80%

All five baseline models show identical degradation patterns, confirming that drift is a **data problem, not a model problem**.


## Pipeline Architecture

### Phase 1: Memory-Safe Preprocessing
- Chunked streaming (20,000 rows per chunk)
- NaN/Inf removal
- Column cleaning (whitespace stripping)
- Label encoding (BENIGN = 0, ATTACK = 1)
- Feature engineering

### Phase 2: Baseline Model Training
- Train on Monday reference dataset (529,481 rows)
- Models:
  - Random Forest  
  - XGBoost  
  - Decision Tree  
  - Logistic Regression  
  - SVM  
- Evaluate on all batches without retraining

### Phase 3: Drift Quantification
- Fixed-reference divergence (Monday vs others)
- Rolling-window divergence
- Feature-level JS divergence heatmaps

### Phase 4: Meta-Model Training
- Linear Regression: (KL, JS, Wasserstein) → Accuracy
- Leave-One-Out Cross-Validation
- Prediction vs actual comparison


## Key Findings

### Model-Agnostic Degradation
All models exhibit identical performance drops under drift.

### Divergence Metrics
KL, JS, and Wasserstein consistently detect major drift events.

### Fixed vs Rolling Drift
- Fixed reference → long-term drift  
- Rolling window → sudden transitions  

### Feature-Level Drift
- DDoS: packet length statistics dominate  
- DoS: window size features dominate  

## SHAP Analysis

- **Decision Tree:** Packet length features amplify  
- **Logistic Regression:** Extreme sensitivity (~17× increase)  
- **Random Forest:** More robust, retains some stability  
- **SVM:** Highest amplification (~25×)  
- **XGBoost:** Feature importance shifts significantly  

## Dataset

**CICIDS 2017 Dataset**

| Batch | Rows | Traffic Type |
|------|------|-------------|
| Monday | 529,481 | Benign |
| Tuesday | 445,645 | Brute Force |
| Wednesday | 691,406 | DoS |
| Thursday AM | 170,231 | Web Attacks |
| Thursday PM | 288,395 | Infiltration |
| Friday AM | 190,911 | Port Scan |
| Friday PM | 286,096 | Aggressive Scan |
| Friday DDoS | 225,711 | DDoS |

**Total:** 2,827,876 rows

## Running the Pipeline

Step 1: Preprocessing
 - python preprocess.py

Step 2: Training
 - python train.py

Step 3: Drift computation
 - python drift.py

Step 4: Meta-model
 - python meta_model.py

Step 5: SHAP analysis (optional)
 - python shap_analysis.py
