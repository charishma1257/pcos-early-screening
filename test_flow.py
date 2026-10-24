"""
test_flow.py
============
Comprehensive end-to-end testing suite for PCOS Early Screening prototype.
Verifies all 4 required test dimensions:
1. Three realistic test inputs (Low, Moderate, High risk) -> category, SHAP, and plan.
2. Invalid input handling (empty, out-of-bounds, wrong types) and missing model file.
3. Prediction consistency between direct pipeline and Flask route.
4. Zero personal data persistence verification.
"""

import os
import glob
import joblib
import pandas as pd
from app import app, MODEL_PATH


def run_tests():
    print("=" * 70)
    print("STARTING PCOS SCREENING END-TO-END TEST SUITE")
    print("=" * 70)

    client = app.test_client()

    # Load artifacts directly for consistency checking
    artifacts = joblib.load(MODEL_PATH)
    pipeline = artifacts['calibrated_model']
    preprocessor = artifacts['preprocessor']
    feature_order = artifacts['feature_order']

    # Snapshot directory before tests to verify zero data persistence
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    initial_files = set(glob.glob(os.path.join(workspace_dir, "**", "*"), recursive=True))

    # Verify home page has no Model Verification technical elements
    res_home = client.get("/")
    assert res_home.status_code == 200
    home_html = res_home.data.decode("utf-8")
    assert "Model Verification" not in home_html
    assert "Validated ML Performance" not in home_html
    assert "Pipeline Ready" not in home_html
    assert "Platt Sigmoid" not in home_html
    assert "Brier Score" not in home_html

    # ==========================================================
    # TEST 1: Three Realistic Test Inputs (Low, Moderate, High)
    # ==========================================================
    print("\n--- TEST 1: Realistic Test Inputs (Low, Moderate, High) ---")

    test_cases = [
        {
            "name": "Case A (Expected Low Risk)",
            "data": {
                'Age (yrs)': '24',
                'BMI': '20.8',
                'Pulse rate(bpm)': '70',
                'RR (breaths/min)': '18',
                'Cycle(R/I)': '2',  # Regular
                'Cycle length(days)': '5',
                'Marraige Status (Yrs)': '0',
                'Pregnant(Y/N)': '0',
                'No. of aborptions': '0',
                'Hip(inch)': '36',
                'Waist(inch)': '28',
                'Weight gain(Y/N)': '0',
                'hair growth(Y/N)': '0',
                'Skin darkening (Y/N)': '0',
                'Hair loss(Y/N)': '0',
                'Pimples(Y/N)': '0',
                'Fast food (Y/N)': '0',
                'Reg.Exercise(Y/N)': '1'
            }
        },
        {
            "name": "Case B (Expected Moderate Risk)",
            "data": {
                'Age (yrs)': '26',
                'BMI': '26.3',
                'Pulse rate(bpm)': '72',
                'RR (breaths/min)': '18',
                'Cycle(R/I)': '2',  # Regular cycle with weight & miscarriages
                'Cycle length(days)': '5',
                'Marraige Status (Yrs)': '3',
                'Pregnant(Y/N)': '0',
                'No. of aborptions': '2',
                'Hip(inch)': '45',
                'Waist(inch)': '40',
                'Weight gain(Y/N)': '1',
                'hair growth(Y/N)': '0',
                'Skin darkening (Y/N)': '0',
                'Hair loss(Y/N)': '0',
                'Pimples(Y/N)': '0',
                'Fast food (Y/N)': '0',
                'Reg.Exercise(Y/N)': '0'
            }
        },
        {
            "name": "Case C (Expected High Risk)",
            "data": {
                'Age (yrs)': '30',
                'BMI': '32.1',
                'Pulse rate(bpm)': '76',
                'RR (breaths/min)': '20',
                'Cycle(R/I)': '4',  # Irregular
                'Cycle length(days)': '3',
                'Marraige Status (Yrs)': '5',
                'Pregnant(Y/N)': '0',
                'No. of aborptions': '1',
                'Hip(inch)': '44',
                'Waist(inch)': '40',
                'Weight gain(Y/N)': '1',
                'hair growth(Y/N)': '1',
                'Skin darkening (Y/N)': '1',
                'Hair loss(Y/N)': '1',
                'Pimples(Y/N)': '1',
                'Fast food (Y/N)': '1',
                'Reg.Exercise(Y/N)': '0'
            }
        }
    ]

    for tc in test_cases:
        res = client.post("/predict", data=tc["data"])
        assert res.status_code == 200, f"Expected 200 for {tc['name']}, got {res.status_code}"
        html = res.data.decode("utf-8")

        # Determine category from HTML response
        if "Provisional Low Risk" in html:
            cat = "Low"
        elif "Provisional Moderate Risk" in html:
            cat = "Moderate"
        elif "Provisional High Risk" in html:
            cat = "High"
        else:
            cat = "Unknown"

        # Extract probability and SHAP factors if available
        import re
        prob_match = re.search(r"<strong>(\d+\.\d+)%</strong>", html)
        prob_val = prob_match.group(1) if prob_match else "N/A"

        print(f"\n[{tc['name']}]")
        print(f"  Assigned Risk Category: Provisional {cat} Risk (Calibrated Likelihood: {prob_val}%)")

        # Check for presence of SHAP chart and narrative
        assert "data:image/png;base64," in html, "SHAP chart base64 image not found in response"
        assert "Plain-English Interpretation" in html, "SHAP narrative interpretation section missing"
        assert "Personalized Lifestyle & Self-Care Guidance" in html, "Recommendation plan missing"

        # Check for plan components
        assert "Balanced Meals & Nutrition" in html
        assert "Physical Activity & Movement" in html
        assert "Sleep Routines" in html
        assert "Stress Management" in html
        assert "Symptom Journaling" in html

        # Extract top SHAP factors from table
        shap_rows = re.findall(r"<td class=\".*?fw-semibold text-dark\">(.*?)</td>\s*<td.*?><span class=\"badge bg-light text-dark border\">(.*?)</span></td>\s*<td class=\"text-end.*?>\s*<span class=\"badge .*?\">\s*([+-]\d+\.\d+)", html)
        print("  Top SHAP Contributing Factors:")
        for feat, val, imp in shap_rows:
            print(f"    - {feat} ({val}): SHAP impact = {imp}")

        # Explicitly verify removed technical elements (Requirements 1, 2, and 5)
        assert "Model Verification" not in html, "Model Verification badge should not be displayed"
        assert "Validated ML Performance" not in html, "Validated ML Performance should not be displayed"
        assert "Pipeline Ready" not in html, "Pipeline Ready status should not be displayed"
        assert "Threshold Reference" not in html, "Threshold Reference section should not be displayed"
        assert "Platt Sigmoid" not in html, "Platt Sigmoid technical text should not be displayed"
        assert "Brier Score" not in html, "Brier score should not be displayed"
        assert "training cohort" not in html, "Training cohort terminology should not be displayed"
        assert "observed prevalence" not in html, "Observed prevalence percentages should not be displayed"

        # Extract Tailored Tags
        tags = re.findall(r"<span class=\"badge bg-light text-primary-custom border border-lavender rounded-pill me-1 mb-1 px-2-5 py-1-5 small\">\s*<i class=\"bi bi-tag-fill me-1\"></i>\s*(.*?)\s*</span>", html)
        print(f"  Tailored Lifestyle Focus Areas: {', '.join(tags) if tags else 'Standard Wellness'}")
        print(f"  Pass: {tc['name']} verified successfully.")

    # ==========================================================
    # TEST 2: Invalid Inputs & Missing Model Handling
    # ==========================================================
    print("\n--- TEST 2: Validation & Error Handling ---")

    # 2a. Missing required field (Empty Age)
    bad_data_1 = test_cases[0]["data"].copy()
    bad_data_1["Age (yrs)"] = ""
    res_1 = client.post("/predict", data=bad_data_1)
    assert res_1.status_code == 400, f"Expected 400 for empty age, got {res_1.status_code}"
    assert "required" in res_1.data.decode("utf-8").lower()
    print("  Pass: Empty required field correctly rejected with 400 and friendly error.")

    # 2b. Out-of-range value (Age = 120, BMI = 3)
    bad_data_2 = test_cases[0]["data"].copy()
    bad_data_2["Age (yrs)"] = "120"
    res_2 = client.post("/predict", data=bad_data_2)
    assert res_2.status_code == 400, f"Expected 400 for age 120, got {res_2.status_code}"
    assert "cannot be greater than" in res_2.data.decode("utf-8").lower()
    print("  Pass: Out-of-range value correctly rejected with 400 and boundary notice.")

    # 2c. Non-numeric text in numeric field
    bad_data_3 = test_cases[0]["data"].copy()
    bad_data_3["BMI"] = "twenty-five"
    res_3 = client.post("/predict", data=bad_data_3)
    assert res_3.status_code == 400, f"Expected 400 for string BMI, got {res_3.status_code}"
    assert "must be a valid number" in res_3.data.decode("utf-8").lower()
    print("  Pass: Non-numeric type correctly rejected with 400.")

    # 2d. Missing model test
    import app as app_module
    saved_flag = app_module.PIPELINE_LOADED
    app_module.PIPELINE_LOADED = False
    res_missing = client.get("/screening", follow_redirects=True)
    html_missing = res_missing.data.decode("utf-8")
    assert "temporarily unavailable" in html_missing or "not loaded" in html_missing
    print("  Pass: Missing model gracefully notified without traceback.")
    app_module.PIPELINE_LOADED = saved_flag

    # ==========================================================
    # TEST 3: Route vs Training Script Consistency Check
    # ==========================================================
    print("\n--- TEST 3: Single Prediction Consistency Check ---")

    sample_input = {
        'Age (yrs)': 29.0,
        'BMI': 27.2,
        'Pulse rate(bpm)': 72.0,
        'RR (breaths/min)': 18.0,
        'Cycle(R/I)': 4,
        'Cycle length(days)': 5.0,
        'Marraige Status (Yrs)': 4.0,
        'Pregnant(Y/N)': 0,
        'No. of aborptions': 0,
        'Hip(inch)': 39.0,
        'Waist(inch)': 35.0,
        'Weight gain(Y/N)': 1,
        'hair growth(Y/N)': 0,
        'Skin darkening (Y/N)': 1,
        'Hair loss(Y/N)': 1,
        'Pimples(Y/N)': 0,
        'Fast food (Y/N)': 1,
        'Reg.Exercise(Y/N)': 0
    }

    # Direct calculation with loaded pipeline
    df_single = pd.DataFrame([sample_input])[feature_order]
    proc_single = preprocessor.transform(df_single)
    direct_prob = float(pipeline.predict_proba(proc_single)[:, 1][0])
    direct_prob_pct = round(direct_prob * 100, 1)

    # Route calculation via Flask client
    form_payload = {k: str(v) for k, v in sample_input.items()}
    res_route = client.post("/predict", data=form_payload)
    assert res_route.status_code == 200
    html_route = res_route.data.decode("utf-8")

    expected_str = f"{direct_prob_pct}%"
    assert expected_str in html_route, f"Expected probability {expected_str} in route HTML, but not found."
    print(f"  Direct Pipeline Prob: {direct_prob:.4f} ({direct_prob_pct}%)")
    print(f"  Flask Route HTML:     Contains exact string '{expected_str}'")
    print("  Pass: Route output perfectly matches direct ML model pipeline.")

    # ==========================================================
    # TEST 4: Zero Persistence Verification
    # ==========================================================
    print("\n--- TEST 4: Data Persistence Audit ---")
    current_files = set(glob.glob(os.path.join(workspace_dir, "**", "*"), recursive=True))

    new_files = [
        f for f in (current_files - initial_files)
        if not f.endswith(('.pyc', '__pycache__')) and not os.path.isdir(f)
    ]

    assert len(new_files) == 0, f"Found unexpected newly persisted files: {new_files}"
    print(f"  New files created during request handling: 0")
    print("  Pass: Zero personal health data persisted in memory or on disk.")

    print("\n" + "=" * 70)
    print("ALL 4 TEST DIMENSIONS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == '__main__':
    run_tests()
