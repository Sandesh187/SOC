#!/usr/bin/env python3
"""
ml_six_models_benchmark.py — 6-Model TBML Machine Learning Benchmark Engine
=============================================================================
Implements and evaluates the six distinct models detailed in the 
"Corporate Compliance: Trade-Based Money Laundering Detection ML Modeling Report":

1. Model 1 (Price Deviation Rule): Over-Invoicing rule (Unit_Price > 1.2 * Market_Spot_Price)
2. Model 2 (Structuring Amounts Rule): Smurfing rule ($9,800 <= Total_Value_USD <= $10,000)
3. Model 3 (High-Risk Jurisdiction Rule): Ghost Shipments / Sanctions (High-risk country list)
4. Model 4 (Combined - Logistic Regression): LogReg trained on flags of Models 1, 2, and 3
5. Model 5 (Pure ML - Random Forest): 100-tree decision ensemble on raw features
6. Model 6 (Pure ML - Gradient Boosting): Sequential boosting classifier on raw features

Outputs:
- Full evaluation table (Accuracy, Precision, Recall, F1-Score) for all 6 models.
- Model artifacts & performance metrics report matching Table 1 of the compliance paper.
"""

import pandas as pd
import numpy as np
import os
import time

def run_benchmark():
    print("=" * 75)
    print(" CORPORATE COMPLIANCE: 6-MODEL MACHINE LEARNING BENCHMARK ENGINE")
    print("=" * 75)

    # 1. Load Data
    data_path = "model_labelled_ledgers.xlsx"
    if not os.path.exists(data_path):
        data_path = "metallurgical_ledgers.xlsx"

    print(f"[*] Loading dataset: '{data_path}'")
    df = pd.read_excel(data_path)
    print(f"[+] Loaded {len(df):,} transaction records.")

    # Define Ground-Truth Fraud / Anomaly Label
    SANCTIONED_COUNTRIES = {
        'Sanctioned_Proxy_Alpha', 'Sanctioned_Proxy_Beta',
        'Iran', 'North Korea', 'Russia', 'Syria'
    }

    if 'Suspicious_Flag' in df.columns:
        y = df['Suspicious_Flag'].values
    else:
        # Ground truth rule definition if label column is raw
        is_over_invoiced = (df['Unit_Price_USD'] > 1.20 * df['Market_Spot_Price'])
        is_structured = (df['Total_Value_USD'] >= 9800) & (df['Total_Value_USD'] <= 10000)
        is_sanctioned = df['Vendor_Country'].isin(SANCTIONED_COUNTRIES)
        y = (is_over_invoiced | is_structured | is_sanctioned).astype(int).values

    # Feature Engineering for Rule-Based Models 1-3
    m1_pred = (df['Unit_Price_USD'] > 1.20 * df['Market_Spot_Price']).astype(int).values
    m2_pred = ((df['Total_Value_USD'] >= 9800) & (df['Total_Value_USD'] <= 10000)).astype(int).values
    m3_pred = df['Vendor_Country'].isin(SANCTIONED_COUNTRIES).astype(int).values

    # Train / Test Split (15,030 train / 3,030 test or 80-20 split)
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)

    y_train, y_test = y[train_idx], y[test_idx]

    # Evaluate Rule-Based Models on Test Set
    m1_test = m1_pred[test_idx]
    m2_test = m2_pred[test_idx]
    m3_test = m3_pred[test_idx]

    results = []

    # Model 1
    results.append({
        "Model": "Model 1: Price Deviation Rule",
        "Accuracy": accuracy_score(y_test, m1_test),
        "Precision": precision_score(y_test, m1_test, zero_division=0),
        "Recall": recall_score(y_test, m1_test, zero_division=0),
        "F1-Score": f1_score(y_test, m1_test, zero_division=0)
    })

    # Model 2
    results.append({
        "Model": "Model 2: Structuring Rule",
        "Accuracy": accuracy_score(y_test, m2_test),
        "Precision": precision_score(y_test, m2_test, zero_division=0),
        "Recall": recall_score(y_test, m2_test, zero_division=0),
        "F1-Score": f1_score(y_test, m2_test, zero_division=0)
    })

    # Model 3
    results.append({
        "Model": "Model 3: High-Risk Country Rule",
        "Accuracy": accuracy_score(y_test, m3_test),
        "Precision": precision_score(y_test, m3_test, zero_division=0),
        "Recall": recall_score(y_test, m3_test, zero_division=0),
        "F1-Score": f1_score(y_test, m3_test, zero_division=0)
    })

    # Model 4: Combined Rule-Based via Logistic Regression
    X_rule_train = np.column_stack((m1_pred[train_idx], m2_pred[train_idx], m3_pred[train_idx]))
    X_rule_test = np.column_stack((m1_test, m2_test, m3_test))

    logreg = LogisticRegression(random_state=42, class_weight='balanced')
    logreg.fit(X_rule_train, y_train)
    m4_test = logreg.predict(X_rule_test)

    results.append({
        "Model": "Model 4: Combined (LogReg)",
        "Accuracy": accuracy_score(y_test, m4_test),
        "Precision": precision_score(y_test, m4_test, zero_division=0),
        "Recall": recall_score(y_test, m4_test, zero_division=0),
        "F1-Score": f1_score(y_test, m4_test, zero_division=0)
    })

    # Raw Feature Preprocessing for Pure ML Models 5 & 6
    raw_feature_cols = ['Volume_MT', 'Market_Spot_Price', 'Unit_Price_USD', 'Total_Value_USD',
                        'Vendor_Country', 'Commodity', 'Payment_Method']
    X_raw = pd.get_dummies(df[raw_feature_cols], drop_first=True)

    X_raw_train = X_raw.iloc[train_idx]
    X_raw_test = X_raw.iloc[test_idx]

    # Model 5: Pure ML (Random Forest)
    print("[*] Training Model 5: Random Forest (100 trees)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X_raw_train, y_train)
    m5_test = rf.predict(X_raw_test)

    results.append({
        "Model": "Model 5: Pure ML (Random Forest)",
        "Accuracy": accuracy_score(y_test, m5_test),
        "Precision": precision_score(y_test, m5_test, zero_division=0),
        "Recall": recall_score(y_test, m5_test, zero_division=0),
        "F1-Score": f1_score(y_test, m5_test, zero_division=0)
    })

    # Model 6: Pure ML (Gradient Boosting)
    print("[*] Training Model 6: Gradient Boosting Classifier...")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=4, subsample=0.8, learning_rate=0.05)
    gb.fit(X_raw_train, y_train)
    m6_test = gb.predict(X_raw_test)

    results.append({
        "Model": "Model 6: Pure ML (Gradient Boosting)",
        "Accuracy": accuracy_score(y_test, m6_test),
        "Precision": precision_score(y_test, m6_test, zero_division=0),
        "Recall": recall_score(y_test, m6_test, zero_division=0),
        "F1-Score": f1_score(y_test, m6_test, zero_division=0)
    })

    # Print Table 1 Results
    df_results = pd.DataFrame(results)
    print("\n" + "=" * 75)
    print(" TABLE 1: MODEL EVALUATION METRICS BENCHMARK")
    print("=" * 75)
    print(df_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 75)

    # Save benchmark table to CSV
    df_results.to_csv("model_evaluation_metrics.csv", index=False)
    print("\n[+] Benchmark results saved to 'model_evaluation_metrics.csv'")

if __name__ == "__main__":
    run_benchmark()
