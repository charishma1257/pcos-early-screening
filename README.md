# PCOS Early Screening and Awareness

A lightweight, functional Flask web prototype for early risk awareness of Polycystic Ovary Syndrome (PCOS). The system takes a non-invasive, patient-friendly questionnaire, evaluates risk using a probability-calibrated XGBoost model, provides transparent feature-contribution explanations via SHAP (TreeExplainer), and outputs adaptive 2-week and 1-month lifestyle recommendation plans.

> **Clinical Disclaimer:** This prototype provides preliminary risk awareness only. It does not diagnose or rule out PCOS. Please consult a qualified healthcare professional.

---

## 1. Project Architecture & Folder Structure

```text
pcos_early_screening/
├── app.py                      # Main Flask application (in-memory routing, error handling)
├── train_model.py              # ML training, probability calibration, and artifact pipeline
├── test_flow.py                # End-to-end automated verification test suite
├── requirements.txt            # Pinned dependency versions
├── README.md                   # System documentation, metrics, and methodology
├── data/
│   └── PCOS_noninvasive1(2).csv # 541-patient non-invasive clinical dataset
├── model/
│   └── pcos_pipeline.joblib    # Unified artifact: preprocessor, calibrated model, SHAP tree model
├── static/
│   ├── css/style.css           # Calm, accessible styling (Plus Jakarta Sans, warm palette)
│   ├── js/main.js              # Client-side validation and interactive BMI calculator
│   └── img/
│       └── calibration_curve.png # Generated probability calibration curve
├── templates/
│   ├── base.html               # Base layout with navbar, footer, and universal disclaimer
│   ├── index.html              # Home page with project overview & model verification metrics
│   ├── questionnaire.html      # 4-section screening form with units and input helpers
│   └── results.html            # Results: Provisional risk tier, SHAP chart, 2-wk/1-mo plan toggle
└── utils/
    ├── __init__.py
    ├── validator.py            # Server-side input validation and range bounds checking
    ├── explainer.py            # SHAP TreeExplainer, narrative builder, and bar chart generator
    └── recommender.py          # Rule-based adaptive lifestyle guidance engine
```

---

## 2. Real Model Metrics (Held-Out Test Data, N=109)

The model was trained on the 541-patient dataset using a stratified 80/20 train/test split (`random_state=42`) with class-imbalance weighting (`scale_pos_weight = 2.06`). The base XGBoost classifier was tuned with 5-fold Stratified cross-validation (`RandomizedSearchCV`) and calibrated using Platt scaling (`CalibratedClassifierCV(method='sigmoid')`).

| Metric | Real Held-Out Test Value |
| :--- | :--- |
| **Accuracy** | **85.32%** (0.8532) |
| **ROC-AUC** | **88.58%** (0.8858) |
| **Precision** | **83.33%** (0.8333) |
| **Recall** | **69.44%** (0.6944) |
| **F1-Score** | **75.76%** (0.7576) |
| **Brier Score** | **0.1195** (calibrated, improved from uncalibrated 0.1328) |

### Test Confusion Matrix (N = 109)

```text
                  Predicted Negative    Predicted Positive
Actual Negative          68 (TN)               5 (FP)
Actual Positive          11 (FN)              25 (TP)
```
- **Specificity:** 93.15% (68 / 73 true negatives identified)
- **Positive Predictive Value (Precision):** 83.33% (25 / 30 positive predictions correct)

---

## 3. Data-Driven Risk Threshold Methodology

To honestly solve the probability threshold problem without fabricating cut-offs:
1. **Out-of-Fold Calibration:** We generated out-of-fold predicted probabilities across the training cohort (N=432) using 5-fold stratified cross-validation.
2. **Prevalence Matching:** The empirical distribution of probabilities revealed distinct prevalence tiers:
   - **Provisional Low Risk (`prob < 0.30`):** 264 training samples fell into this bucket. Observed PCOS prevalence in this cohort was only **10.23%** (giving an empirical negative predictive value of ~90%).
   - **Provisional Moderate Risk (`0.30 <= prob < 0.60`):** 66 training samples fell into this intermediate zone. Observed PCOS prevalence was **43.94%**, representing genuine clinical uncertainty requiring watchful monitoring.
   - **Provisional High Risk (`prob >= 0.60`):** 102 training samples fell into this bucket. Observed PCOS prevalence was **83.33%**, providing high positive predictive value.
3. **Provisional Labeling:** All categories are explicitly presented as **"Provisional"** in the user interface to communicate that machine-learning screening is not an official medical diagnosis.

---

## 4. Explainable AI (SHAP)

- Explanations are calculated per patient in real-time using `shap.TreeExplainer` on the fitted tree model.
- The UI presents a horizontal bar chart displaying the top 5 contributors with distinct color codes:
  - **Red / Coral (`#dc3545`):** Positive SHAP contribution (factors that increased the model's estimate).
  - **Green / Teal (`#198754`):** Negative SHAP contribution (factors that decreased the model's estimate).
- Plain-English narratives are constructed dynamically using careful clinical language (*"contributed to the model's estimate"*, avoiding words like *"caused"* or *"proves"*).
- Every explanation includes a note emphasizing that SHAP explains the statistical behavior of the algorithm, not biological causality.

---

## 5. Adaptive Lifestyle Recommendation Engine

The results page features an interactive **2-Week vs. 1-Month** plan selector that adapts according to:
- **Assigned Risk Tier:** High-risk plans emphasize immediate consultation with a gynecologist or endocrinologist alongside supportive wellness habits. Low/Moderate tiers provide preventive routines with a medical evaluation reminder if symptoms persist.
- **Specific Questionnaire Answers:**
  - *Fast food consumption:* Triggers low-GI dietary swaps, blood sugar stabilization strategies, and anti-inflammatory whole foods.
  - *Exercise status:* Tailors physical activity between gentle low-impact walks (if sedentary) or strength/recovery balancing (if already active).
  - *Cycle regularity:* Provides targeted menstrual and basal temperature tracking protocols.
  - *BMI & Metabolic symptoms (acanthosis/weight gain):* Adds insulin sensitivity guidance.
- **Medical Guardrails:** Strictly excludes medication advice, makes no promises of curing or treating PCOS, and provides general non-clinical wellness support only.

---

## 6. Installation & Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- pip

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: (Re)Train Model & Generate Artifacts
```bash
python train_model.py
```
This trains the calibrated XGBoost model on `data/PCOS_noninvasive1(2).csv`, prints held-out metrics, generates the calibration plot, and saves `model/pcos_pipeline.joblib`.

### Step 3: Run the Web App
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your web browser.

### Step 4: Run the Automated Verification Suite
```bash
python test_flow.py
```
Verifies all 4 test dimensions:
1. Low, Moderate, and High test inputs run through Flask.
2. Validation of empty, out-of-range, and wrong-type inputs, plus missing model handling.
3. Prediction consistency between direct pipeline and Flask route.
4. Audit confirming zero patient data persistence.

---

## 7. Data Privacy Guarantee

- **Zero Persistence:** No database is connected. No input forms are written to disk, SQLite, or log files.
- **In-Memory Only:** Questionnaire inputs are evaluated in volatile memory and discarded immediately after the HTTP response is rendered.

---

## 8. Known Limitations

1. **Screening Tool Only:** PCOS diagnosis requires the Rotterdam criteria (at least two of: oligo/anovulation, clinical/biochemical hyperandrogenism, and polycystic ovarian morphology via ultrasound). A questionnaire cannot confirm or exclude PCOS without laboratory and imaging diagnostics.
2. **Vital Signs Self-Reporting:** Pulse rate and respiratory rate are clinical vitals. While the form provides standard population median defaults (72 bpm, 18 breaths/min), patient self-measurement may introduce variance.
3. **Sample Size:** The dataset comprises 541 patients from a single clinical cohort. External multi-center validation on diverse demographic populations is recommended before clinical translation.
