#!/usr/bin/env python3
"""
predict_fraud.py — TBML Prediction & Labelling Script
======================================================
Loads the unlabelled 'metallurgical_ledgers (1).csv', applies the ML model
(Gradient Boosting) and heuristic rules to predict whether a transaction is 
fraudulent, and identifies the specific Fraud_Type.

Outputs a fully labelled Excel file: 'predicted_metallurgical_ledgers.xlsx'
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

SANCTIONED_COUNTRIES = {'Sanctioned_Proxy_Alpha', 'Sanctioned_Proxy_Beta', 'Iran', 'North Korea', 'Russia', 'Syria'}
PRICE_DEVIATION_RATIO = 1.20
STRUCTURING_MIN_USD = 9800
STRUCTURING_MAX_USD = 10000
def generate_predictions():
    print("=" * 70)
    print(" 1. LOADING DATA FOR PREDICTION")
    print("=" * 70)
    
    input_file = "metallurgical_ledgers (1).csv"
    if not os.path.exists(input_file):
        print(f"[!] Error: {input_file} not found.")
        return

    df = pd.read_csv(input_file)
    print(f"[*] Loaded {len(df):,} rows from {input_file}")

    # BUG FIX: previously, real ground-truth labels (if the input file had
    # them) were renamed straight into 'Is_Fraud_C' -- but that column gets
    # unconditionally overwritten by the model's own predictions a few lines
    # below, so the real labels were silently discarded and never used for
    # anything. Now we keep them in a separate column and actually use them
    # to report how well the model agrees with reality, if they exist.
    has_ground_truth = 'Is_Fraud_Ground_Truth' in df.columns
    if has_ground_truth:
        ground_truth = df['Is_Fraud_Ground_Truth'].astype(int)
    elif 'Is_Fraud_C' in df.columns:
        # Column already looks like a label column from a prior run; treat
        # it as ground truth for evaluation purposes too, but don't let it
        # block re-prediction below.
        has_ground_truth = True
        ground_truth = df['Is_Fraud_C'].astype(int)
    else:
        ground_truth = None

    df['Is_Fraud_C'] = 0
    if 'Fraud_Type' not in df.columns:
        df['Fraud_Type'] = 'None'

    print("\n" + "=" * 70)
    print(" 2. DETERMINING FRAUD TYPES (TYPOLOGIES)")
    print("=" * 70)

    # Calculate rule-based flags (using the shared, single-source-of-truth
    # thresholds from tbml_constants so this file can't drift out of sync
    # with financial_model.py / ml_six_models_benchmark.py / the SQL pipeline)
    is_over_invoiced = (df['Unit_Price_USD'] > PRICE_DEVIATION_RATIO * df['Market_Spot_Price'])
    is_structured = (df['Total_Value_USD'] >= STRUCTURING_MIN_USD) & (df['Total_Value_USD'] <= STRUCTURING_MAX_USD)
    is_sanctioned = df['Vendor_Country'].isin(SANCTIONED_COUNTRIES)

    # Target variable for training the ML Model.
    # NOTE: if no independent ground truth is available, this rule-based OR
    # is used as a training target -- which means the model is learning a
    # smoothed version of these exact three rules, not discovering fraud
    # independently. Train/predict also happen on the same rows (no
    # holdout), so predictions below are in-sample fitted values, not a
    # generalization test. This script is a labelling tool, not a benchmark
    # -- see ml_six_models_benchmark.py for held-out evaluation.
    y_target = (is_over_invoiced | is_structured | is_sanctioned).astype(int)

    print("\n" + "=" * 70)
    print(" 3. TRAINING ML MODEL (GRADIENT BOOSTING) & PREDICTING")
    print("=" * 70)
    
    raw_feature_cols = ['Volume_MT', 'Market_Spot_Price', 'Unit_Price_USD', 'Total_Value_USD',
                        'Vendor_Country', 'Commodity', 'Payment_Method']
    
    print("[*] Preprocessing features...")
    X_raw = pd.get_dummies(df[raw_feature_cols], drop_first=True)
    
    print("[*] Training Gradient Boosting Classifier (Model 6)...")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=4, subsample=0.8, learning_rate=0.05)
    gb.fit(X_raw, y_target)
    
    print("[*] Generating Predictions...")
    df['Is_Fraud_C'] = gb.predict(X_raw)

    if has_ground_truth:
        print("\n[*] Ground-truth column detected -- evaluating predictions against it")
        print("    (NOTE: model was trained and predicted on these same rows,")
        print("     so this is an in-sample fit check, not a generalization test):")
        print(f"      Accuracy : {accuracy_score(ground_truth, df['Is_Fraud_C']):.4f}")
        print(f"      Precision: {precision_score(ground_truth, df['Is_Fraud_C'], zero_division=0):.4f}")
        print(f"      Recall   : {recall_score(ground_truth, df['Is_Fraud_C'], zero_division=0):.4f}")
        print(f"      F1-Score : {f1_score(ground_truth, df['Is_Fraud_C'], zero_division=0):.4f}")

    print("\n" + "=" * 70)
    print(" 4. ASSIGNING FRAUD TYPES")
    print("=" * 70)
    
    # BUG FIX: the old version assigned Fraud_Type via three sequential
    # `.loc[]` writes (over-invoicing -> structuring -> sanction). A
    # transaction matching more than one rule was silently overwritten by
    # whichever assignment ran last (sanction always won), with no
    # indication that the other typologies also applied. For a compliance
    # tool, hiding co-occurring typologies is a real loss of information --
    # a transaction that is BOTH over-invoiced AND sanctioned is more
    # suspicious than either alone, and an investigator should see that.
    # Now every matching typology is reported, joined together.
    mask_fraud = (df['Is_Fraud_C'] == 1)

    typology_matrix = pd.DataFrame({
        'Over-Invoicing':     mask_fraud & is_over_invoiced,
        'Structuring':        mask_fraud & is_structured,
        'Sanction Violation': mask_fraud & is_sanctioned,
    })

    def _combine_typologies(row):
        hits = [name for name, matched in row.items() if matched]
        if hits:
            return ' + '.join(hits)
        return 'Complex ML Anomaly'  # flagged by the model, no rule matched

    df['Fraud_Type'] = 'None'
    df.loc[mask_fraud, 'Fraud_Type'] = typology_matrix.loc[mask_fraud].apply(_combine_typologies, axis=1)
    
    fraud_count = df['Is_Fraud_C'].sum()
    print(f"[+] Predicted {fraud_count:,} fraudulent transactions.")
    print("\nFraud Type Breakdown:")
    print(df[df['Is_Fraud_C'] == 1]['Fraud_Type'].value_counts().to_string())

    print("\n" + "=" * 70)
    print(" 5. EXPORTING RESULTS")
    print("=" * 70)
    
    output_excel = "predicted_metallurgical_ledgers.xlsx"
    df.to_excel(output_excel, index=False)
    print(f"[+] Predictions saved successfully to '{output_excel}'")

if __name__ == "__main__":
    generate_predictions()