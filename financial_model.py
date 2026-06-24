#!/usr/bin/env python3
"""
financial_model.py — Combined Financial Anomaly Detection Model
================================================================
Integrates all learnings from Weeks 1–5 of the Summer of Code programme
into a multi-factor suspicious-transaction detection engine.

Week 1  →  Yahoo Finance & UN Comtrade data concepts (market prices, trade flows)
Week 2  →  World Bank governance scores, shipment delays, smurfing, Benford's Law
Week 3  →  OFAC sanctions screening, rule-based transaction labelling
Week 4  →  Least-squares regression (normal equation β = (XᵀX)⁻¹Xᵀy)
Week 5  →  Skellam & Tweedie distributions, statistical modelling

Improvements over the Week 3 baseline:
  1. Dynamic pricing model (least-squares) replaces the static 8% deviation check
  2. Benford's Law chi-square test replaces the naive round-number flag
  3. Smurfing detection uses statistical clustering instead of a fixed count threshold
  4. Transaction velocity analysis catches temporal anomalies
  5. Mahalanobis distance catches multivariate outliers invisible to univariate rules
  6. Weighted probability fusion replaces the unweighted binary flag sum

Usage:
    python financial_model.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
import time
import warnings

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════════════
# MODULE 1 — OFAC & Jurisdiction Risk Screening  (Weeks 2 + 3)
# ════════════════════════════════════════════════════════════════════════════════

class OFACScreener:
    """
    Screens vendor countries against OFAC sanctions lists and assigns a
    continuous jurisdiction risk score informed by World Bank governance tiers.

    Week 3 improvement
    ──────────────────
    Instead of a binary 0/1 flag, we also output a *continuous* jurisdiction
    score (0.0 – 1.0) so that partial risk from high-risk-but-not-sanctioned
    countries (e.g. UAE as a known transhipment hub) is captured by the
    fusion engine downstream.
    """

    # Directly sanctioned or proxy-sanctioned countries
    SANCTIONED = {
        "Sanctioned_Proxy_Alpha",
        "Sanctioned_Proxy_Beta",
        "Iran",
        "North Korea",
        "Russia",
        "Syria",
        "Cuba",
    }

    # World Bank governance-informed risk tiers (Week 2 learning)
    # 0.0 = very low risk → 1.0 = sanctioned / highest risk
    JURISDICTION_RISK = {
        "Switzerland":              0.05,
        "Germany":                  0.08,
        "Canada":                   0.08,
        "Japan":                    0.07,
        "Australia":                0.06,
        "United States":            0.10,
        "Singapore":                0.09,
        "United Arab Emirates":     0.35,   # known transhipment hub
        "Sanctioned_Proxy_Alpha":   1.00,
        "Sanctioned_Proxy_Beta":    1.00,
    }
    DEFAULT_RISK = 0.50  # fallback for unknown countries

    def screen(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add OFAC binary flag and continuous jurisdiction risk score."""
        out = df.copy()

        # Binary OFAC flag
        out["OFAC_Flag"] = out["Vendor_Country"].isin(self.SANCTIONED).astype(int)

        # Continuous jurisdiction risk
        out["Jurisdiction_Risk"] = (
            out["Vendor_Country"]
            .map(self.JURISDICTION_RISK)
            .fillna(self.DEFAULT_RISK)
        )

        flagged = out["OFAC_Flag"].sum()
        print(f"  [OFAC] Sanctioned-country transactions: {flagged} / {len(out)}")
        return out


# ════════════════════════════════════════════════════════════════════════════════
# MODULE 2 — Least-Squares Pricing Model  (Weeks 1 + 4)
# ════════════════════════════════════════════════════════════════════════════════

class PricingModel:
    """
    Per-commodity ordinary least-squares regression to predict the *fair*
    unit price from observable market features.

    The Normal Equation from Week 4
    ────────────────────────────────
        β = (XᵀX)⁻¹ Xᵀy

    Features (design matrix X):
      - Market_Spot_Price       (from Yahoo Finance / Week 1)
      - log(1 + Volume_MT)     (captures volume discounts)
      - Payment_Method dummies  (Letter_of_Credit, Open_Account; Wire = baseline)

    Week 3 improvement
    ──────────────────
    Replaces the static 8 % deviation threshold with a per-commodity z-score
    that adapts dynamically to each commodity's inherent price scatter.
    """

    def __init__(self, zscore_threshold: float = 2.5):
        self.zscore_threshold = zscore_threshold
        self.models: dict = {}

    # ── helper: build feature matrix ──────────────────────────────────────────
    def _encode_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build the augmented feature matrix [1 | X] for one commodity group."""
        X = np.column_stack([
            df["Market_Spot_Price"].values,
            np.log1p(df["Volume_MT"].values),
            (df["Payment_Method"] == "Letter_of_Credit").astype(float).values,
            (df["Payment_Method"] == "Open_Account").astype(float).values,
        ])
        m = X.shape[0]
        return np.c_[np.ones(m), X]          # prepend bias column of 1s

    # ── main method ───────────────────────────────────────────────────────────
    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit a model per commodity and append prediction columns."""
        out = df.copy()
        out["Predicted_Price"] = np.nan
        out["Price_Residual"]  = np.nan
        out["Price_ZScore"]    = np.nan
        out["Price_Anomaly"]   = 0

        for commodity in df["Commodity"].unique():
            mask = df["Commodity"] == commodity
            sub  = df[mask]

            X_b = self._encode_features(sub)
            y   = sub["Unit_Price_USD"].values

            # ── Normal Equation: θ = (XᵀX)⁻¹ Xᵀy ──
            XtX = X_b.T @ X_b
            det = np.linalg.det(XtX)
            XtX_inv = np.linalg.pinv(XtX) if abs(det) < 1e-10 else np.linalg.inv(XtX)

            theta = XtX_inv @ (X_b.T @ y)

            # Predictions & residuals
            y_pred    = X_b @ theta
            residuals = y - y_pred
            sigma     = np.std(residuals)

            # R² score
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2     = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Z-scores
            zscores = residuals / sigma if sigma > 0 else np.zeros_like(residuals)

            # Store model for reference
            self.models[commodity] = {
                "theta": theta,
                "sigma": sigma,
                "r2":    r2,
                "features": ["bias", "Spot_Price", "log_Volume", "LC_dummy", "OA_dummy"],
            }

            # Write back
            idx = sub.index
            out.loc[idx, "Predicted_Price"] = y_pred
            out.loc[idx, "Price_Residual"]  = residuals
            out.loc[idx, "Price_ZScore"]    = zscores
            out.loc[idx, "Price_Anomaly"]   = (np.abs(zscores) > self.zscore_threshold).astype(int)

            anomalies = (np.abs(zscores) > self.zscore_threshold).sum()
            print(
                f"  [Pricing] {commodity:18s}  R2={r2:.4f}  sigma={sigma:.4f}  "
                f"anomalies={anomalies}/{len(sub)} (|z|>{self.zscore_threshold})"
            )

        return out


# ════════════════════════════════════════════════════════════════════════════════
# MODULE 3 — Benford's Law Analyzer  (Week 2)
# ════════════════════════════════════════════════════════════════════════════════

class BenfordAnalyzer:
    """
    Applies Benford's Law to detect fabricated or manipulated transaction values.

    Benford's Law
    ─────────────
        P(d) = log₁₀(1 + 1/d)   for d ∈ {1, 2, … , 9}

    For each vendor we extract the leading digit of Total_Value_USD across
    all their transactions and run a χ² goodness-of-fit test.

    Week 3 improvement
    ──────────────────
    Replaces the simplistic "Total_Value_USD mod 1000 == 0" round-number
    flag with a statistically grounded first-digit distribution test that
    catches a much wider class of number fabrication.
    """

    # Expected first-digit probabilities under Benford's Law
    EXPECTED = np.array([np.log10(1 + 1 / d) for d in range(1, 10)])

    def __init__(self, significance: float = 0.05, min_transactions: int = 20):
        self.significance = significance
        self.min_transactions = min_transactions

    @staticmethod
    def _first_digit(x: float) -> int:
        """Extract the leading (first) digit of a positive number."""
        if x <= 0:
            return 0
        s = f"{abs(x):.10e}"
        return int(s[0])

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run per-vendor Benford analysis and append result columns."""
        out = df.copy()
        out["Benford_ChiSq"]   = 0.0
        out["Benford_PValue"]  = 1.0
        out["Benford_Anomaly"] = 0

        vendors_tested  = 0
        vendors_flagged = 0

        for vendor_id, group in df.groupby("Vendor_ID"):
            idx    = group.index
            values = group["Total_Value_USD"].values

            if len(values) < self.min_transactions:
                continue

            # Extract first digits
            digits = np.array([self._first_digit(v) for v in values])
            digits = digits[(digits >= 1) & (digits <= 9)]

            if len(digits) < self.min_transactions:
                continue

            # Observed vs expected frequencies
            observed = np.array([(digits == d).sum() for d in range(1, 10)]).astype(float)
            expected = self.EXPECTED * len(digits)

            # χ² statistic  (df = 9 − 1 = 8)
            nonzero = expected > 0
            chi2 = np.sum(
                (observed[nonzero] - expected[nonzero]) ** 2 / expected[nonzero]
            )
            p_value = 1 - stats.chi2.cdf(chi2, df=8)

            vendors_tested += 1
            is_anomaly = int(p_value < self.significance)
            if is_anomaly:
                vendors_flagged += 1

            out.loc[idx, "Benford_ChiSq"]   = chi2
            out.loc[idx, "Benford_PValue"]   = p_value
            out.loc[idx, "Benford_Anomaly"]  = is_anomaly

        print(
            f"  [Benford] Vendors tested: {vendors_tested}, "
            f"flagged: {vendors_flagged} (p < {self.significance})"
        )
        return out


# ════════════════════════════════════════════════════════════════════════════════
# MODULE 4 — Smurfing / Structuring Detector  (Week 2)
# ════════════════════════════════════════════════════════════════════════════════

class SmurfingDetector:
    """
    Detects transaction structuring (smurfing) patterns.

    Smurfing = breaking a large value into many small transactions to stay
    below regulatory reporting thresholds.

    Week 3 improvement
    ──────────────────
    * Uses *coefficient of variation* (CV) of transaction amounts per vendor
      instead of a fixed-count threshold.  Low CV ⇒ suspiciously uniform
      amounts, a hallmark of structuring.
    * Detects *threshold-avoidance clustering*: amounts that cluster just
      below common round thresholds (1 K, 5 K, 10 K, 50 K, 100 K).
    * *Temporal burst detection*: many transactions packed into a short
      calendar window.
    """

    def __init__(self, burst_window_days: int = 7, burst_threshold: int = 5):
        self.burst_window_days = burst_window_days
        self.burst_threshold   = burst_threshold

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run smurfing detection and append columns."""
        out = df.copy()
        out["Smurf_CV_Score"]        = 0.0
        out["Smurf_Burst_Score"]     = 0.0
        out["Smurf_Threshold_Score"] = 0.0
        out["Smurf_Combined"]        = 0.0
        out["Smurf_Anomaly"]         = 0

        out["Date"] = pd.to_datetime(out["Date"])

        for vendor_id, group in df.groupby("Vendor_ID"):
            idx     = group.index
            amounts = group["Total_Value_USD"].values
            dates   = pd.to_datetime(group["Date"])

            # ── 1. Coefficient-of-variation score ─────────────────────────
            # Legitimate vendors show natural variation in transaction sizes.
            # Smurfers tend to use very similar amounts  →  low CV.
            mean_amt = np.mean(amounts)
            std_amt  = np.std(amounts)
            cv       = std_amt / mean_amt if mean_amt > 0 else 1.0
            # Map CV → [0, 1]:  CV < 0.1 → score ≈ 1;  CV ≥ 0.5 → score = 0
            cv_score = float(np.clip(1.0 - cv / 0.5, 0, 1))

            # ── 2. Temporal burst score ───────────────────────────────────
            # Count maximum transactions within any rolling 7-day window.
            dates_sorted = dates.sort_values().reset_index(drop=True)
            max_burst = 1
            j = 0
            for i in range(len(dates_sorted)):
                window_end = dates_sorted.iloc[i] + pd.Timedelta(
                    days=self.burst_window_days
                )
                while j < len(dates_sorted) and dates_sorted.iloc[j] <= window_end:
                    j += 1
                max_burst = max(max_burst, j - i)

            # Normalise: a burst of ≥ burst_threshold in the window is fully suspicious
            if self.burst_threshold > 1:
                burst_score = float(
                    np.clip((max_burst - 2) / (self.burst_threshold - 2), 0, 1)
                )
            else:
                burst_score = 0.0

            # ── 3. Threshold-avoidance score ──────────────────────────────
            # Check if amounts cluster just *below* common round thresholds.
            thresholds = [1_000, 5_000, 10_000, 50_000, 100_000]
            near_count = 0
            for amt in amounts:
                for t in thresholds:
                    remainder = amt % t
                    ratio = remainder / t
                    if 0.85 <= ratio <= 0.999:
                        near_count += 1
                        break  # count once per transaction
            threshold_score = float(
                np.clip(near_count / max(len(amounts) * 0.15, 1), 0, 1)
            )

            # ── Combined score (weighted average) ─────────────────────────
            combined = 0.40 * cv_score + 0.30 * burst_score + 0.30 * threshold_score

            out.loc[idx, "Smurf_CV_Score"]        = cv_score
            out.loc[idx, "Smurf_Burst_Score"]     = burst_score
            out.loc[idx, "Smurf_Threshold_Score"] = threshold_score
            out.loc[idx, "Smurf_Combined"]        = round(combined, 4)
            out.loc[idx, "Smurf_Anomaly"]         = int(combined > 0.50)

        flagged = out["Smurf_Anomaly"].sum()
        print(f"  [Smurfing] Flagged transactions: {flagged} / {len(out)}")
        return out


# ════════════════════════════════════════════════════════════════════════════════
# MODULE 5 — Transaction Velocity Analyzer  (Week 2 delay concepts)
# ════════════════════════════════════════════════════════════════════════════════

class TransactionVelocityAnalyzer:
    """
    Analyses temporal patterns to detect unusual transaction velocity.

    Since the dataset does not contain shipment dates, we adapt the Week 2
    deal-to-shipment delay concepts to *transaction timing*:
      • Inter-arrival time analysis per vendor
      • Same-day transaction clustering
      • Overall transaction cadence anomalies

    Week 2 connection
    ─────────────────
    The delay-risk model taught us that *timing deviations* from expected
    patterns are strong TBML indicators.  Here we apply the same principle
    to the time gaps between a vendor's successive transactions.
    """

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-vendor velocity anomaly scores."""
        out = df.copy()
        out["Date"] = pd.to_datetime(out["Date"])
        out["Velocity_Score"]   = 0.0
        out["Velocity_Anomaly"] = 0

        for vendor_id, group in df.groupby("Vendor_ID"):
            idx   = group.index
            dates = pd.to_datetime(group["Date"]).sort_values()

            if len(dates) < 3:
                continue

            # ── Inter-arrival time statistics ─────────────────────────────
            diffs = dates.diff().dt.days.dropna().values.astype(float)
            if len(diffs) == 0:
                continue

            mean_gap = np.mean(diffs)

            # Fraction of same-day (gap == 0) consecutive transactions
            zero_gap_frac = np.mean(diffs == 0)

            # High fraction of same-day transactions is suspicious
            velocity_score = float(np.clip(zero_gap_frac * 3.0, 0, 1))

            # Also flag hyper-active vendors (mean gap < 3 days over many txns)
            if mean_gap < 3.0 and len(dates) > 20:
                velocity_score = max(velocity_score, 0.5)

            out.loc[idx, "Velocity_Score"]   = round(velocity_score, 4)
            out.loc[idx, "Velocity_Anomaly"] = int(velocity_score > 0.40)

        flagged = out["Velocity_Anomaly"].sum()
        print(f"  [Velocity] Flagged transactions: {flagged} / {len(out)}")
        return out


# ════════════════════════════════════════════════════════════════════════════════
# MODULE 6 — Statistical Outlier Detector  (Week 5)
# ════════════════════════════════════════════════════════════════════════════════

class StatisticalOutlierDetector:
    """
    Uses the *Mahalanobis distance* to detect multivariate outliers within
    each commodity group.

    Mahalanobis distance
    ────────────────────
        D² = (x − μ)ᵀ Σ⁻¹ (x − μ)

    Under multivariate normality D² ~ χ²(p), which gives us a principled
    p-value for each observation.

    Week 5 connection
    ─────────────────
    The Tweedie distribution's handling of zero-inflated continuous data
    inspired using robust statistical methods rather than simple
    percentile-based cut-offs.  Mahalanobis distance accounts for
    *correlations between features*, catching anomalies that would be
    invisible to any single univariate check.
    """

    def __init__(self, significance: float = 0.005):
        self.significance = significance

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Mahalanobis distance per commodity group."""
        out = df.copy()
        out["Mahalanobis_Dist"]   = 0.0
        out["Mahalanobis_PValue"] = 1.0
        out["Statistical_Outlier"] = 0

        for commodity in df["Commodity"].unique():
            mask = df["Commodity"] == commodity
            sub  = df.loc[mask]
            idx  = sub.index

            # Build feature matrix (log-transform the skewed columns)
            X = np.column_stack([
                np.log1p(sub["Volume_MT"].values),
                sub["Unit_Price_USD"].values,
                np.log1p(sub["Total_Value_USD"].values),
            ])

            mu  = np.mean(X, axis=0)
            cov = np.cov(X, rowvar=False)
            cov += np.eye(cov.shape[0]) * 1e-6          # regularisation

            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov)

            # D² for each observation
            diff = X - mu
            d_sq = np.sum(diff @ cov_inv * diff, axis=1)
            d_sq = np.maximum(d_sq, 0.0)                # numerical safety

            # p-value from χ²(p) distribution
            p_features = X.shape[1]
            p_values   = 1 - stats.chi2.cdf(d_sq, df=p_features)

            out.loc[idx, "Mahalanobis_Dist"]    = np.sqrt(d_sq)
            out.loc[idx, "Mahalanobis_PValue"]  = p_values
            out.loc[idx, "Statistical_Outlier"] = (p_values < self.significance).astype(int)

            outliers = (p_values < self.significance).sum()
            print(
                f"  [Outlier] {commodity:18s}  "
                f"outliers={outliers}/{len(sub)} (p < {self.significance})"
            )

        return out


# ════════════════════════════════════════════════════════════════════════════════
# FUSION ENGINE — Combined Financial Anomaly Model
# ════════════════════════════════════════════════════════════════════════════════

class FinancialAnomalyModel:
    """
    Orchestrates all six detection modules and fuses their outputs into a
    single **Unified Suspicion Index** (0–100).

    Fusion formula
    ──────────────
        USI = ( w₁·S_jurisdiction  +  w₂·S_pricing  +  w₃·S_benford
              + w₄·S_smurfing  +  w₅·S_velocity  +  w₆·S_outlier ) × 100

    Each component score S is normalised to [0, 1] before fusion.
    Transactions with USI ≥ threshold are labelled **Suspicious**.
    """

    DEFAULT_WEIGHTS = {
        "jurisdiction": 0.25,    # OFAC + governance risk
        "pricing":      0.25,    # Least-squares price anomaly
        "benford":      0.15,    # Benford's Law deviation
        "smurfing":     0.15,    # Structuring patterns
        "velocity":     0.10,    # Temporal anomalies
        "statistical":  0.10,    # Multivariate outliers
    }

    def __init__(self, weights: dict = None, suspicious_threshold: float = 35.0):
        self.weights              = weights or self.DEFAULT_WEIGHTS
        self.suspicious_threshold = suspicious_threshold

        # Instantiate all modules
        self.ofac     = OFACScreener()
        self.pricing  = PricingModel(zscore_threshold=2.5)
        self.benford  = BenfordAnalyzer(significance=0.05, min_transactions=20)
        self.smurfing = SmurfingDetector(burst_window_days=7, burst_threshold=5)
        self.velocity = TransactionVelocityAnalyzer()
        self.outlier  = StatisticalOutlierDetector(significance=0.005)

    # ── main pipeline ─────────────────────────────────────────────────────────
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full six-module detection pipeline and fuse the results."""

        print("======================================================================")
        print("  COMBINED FINANCIAL ANOMALY DETECTION MODEL")
        print(
            f"  Transactions: {len(df):,}  |  "
            f"Vendors: {df['Vendor_ID'].nunique()}  |  "
            f"Countries: {df['Vendor_Country'].nunique()}"
        )
        print("======================================================================")

        # ── Stage 1–6: run every module sequentially ──────────────────────
        print("\n[Stage 1/7] OFAC & Jurisdiction Screening ...")
        result = self.ofac.screen(df)

        print("\n[Stage 2/7] Least-Squares Pricing Model ...")
        result = self.pricing.fit_predict(result)

        print("\n[Stage 3/7] Benford's Law Analysis ...")
        result = self.benford.analyze(result)

        print("\n[Stage 4/7] Smurfing Detection ...")
        result = self.smurfing.detect(result)

        print("\n[Stage 5/7] Transaction Velocity Analysis ...")
        result = self.velocity.analyze(result)

        print("\n[Stage 6/7] Statistical Outlier Detection ...")
        result = self.outlier.detect(result)

        # ── Stage 7: normalise & fuse ─────────────────────────────────────
        print("\n[Stage 7/7] Fusing scores -> Unified Suspicion Index ...")

        w = self.weights

        # S_jurisdiction: already in [0, 1]
        S_juris = result["Jurisdiction_Risk"]

        # S_pricing: map |z-score| -> [0, 1]  (cap at 5 sigma)
        S_price = np.clip(np.abs(result["Price_ZScore"]) / 5.0, 0, 1)

        # S_benford: lower p-value -> higher suspicion
        S_benf = 1.0 - result["Benford_PValue"]

        # S_smurfing: already in [0, 1]
        S_smurf = result["Smurf_Combined"]

        # S_velocity: already in [0, 1]
        S_vel = result["Velocity_Score"]

        # S_statistical: lower p-value → higher suspicion
        S_stat = 1.0 - result["Mahalanobis_PValue"]

        result["Suspicion_Index"] = (
            w["jurisdiction"] * S_juris
            + w["pricing"]   * S_price
            + w["benford"]   * S_benf
            + w["smurfing"]  * S_smurf
            + w["velocity"]  * S_vel
            + w["statistical"] * S_stat
        ) * 100.0

        result["Suspicion_Index"] = result["Suspicion_Index"].round(2)

        # ── Final label ───────────────────────────────────────────────────
        result["Suspicious_Label"] = result["Suspicion_Index"].apply(
            lambda x: "Suspicious" if x >= self.suspicious_threshold else "Non-Suspicious"
        )

        # ── Summary statistics ────────────────────────────────────────────
        n_sus = (result["Suspicious_Label"] == "Suspicious").sum()
        print(f"\n{'=' * 70}")
        print("  RESULTS SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Total transactions analysed :  {len(result):,}")
        print(
            f"  Suspicious  (USI >= {self.suspicious_threshold})      :  "
            f"{n_sus:,}  ({n_sus / len(result) * 100:.1f} %)"
        )
        print(f"  Non-suspicious              :  {len(result) - n_sus:,}")
        print(f"  Mean   Suspicion Index       :  {result['Suspicion_Index'].mean():.2f}")
        print(f"  Median Suspicion Index       :  {result['Suspicion_Index'].median():.2f}")
        print(f"  Max    Suspicion Index       :  {result['Suspicion_Index'].max():.2f}")
        print(f"\n  Per-module flag breakdown:")
        print(f"    OFAC Flag            :  {result['OFAC_Flag'].sum():>6,}")
        print(f"    Price Anomaly        :  {result['Price_Anomaly'].sum():>6,}")
        print(f"    Benford Anomaly      :  {result['Benford_Anomaly'].sum():>6,}")
        print(f"    Smurfing Anomaly     :  {result['Smurf_Anomaly'].sum():>6,}")
        print(f"    Velocity Anomaly     :  {result['Velocity_Anomaly'].sum():>6,}")
        print(f"    Statistical Outlier  :  {result['Statistical_Outlier'].sum():>6,}")

        return result

    # ── comparison with the Week 3 baseline ───────────────────────────────────
    def compare_with_week3(self, result: pd.DataFrame):
        """Recreate the Week 3 labelling logic and compare overlap."""

        print(f"\n{'=' * 70}")
        print("  COMPARISON WITH WEEK 3 BASELINE")
        print(f"{'=' * 70}")

        df = result

        # ── Recreate Week 3 criteria inline ───────────────────────────────
        sanctioned_countries = [
            "Sanctioned_Proxy_Alpha", "Sanctioned_Proxy_Beta",
            "Iran", "North Korea", "Russia", "Syria", "Cuba",
        ]
        w3_ofac  = df["Vendor_Country"].isin(sanctioned_countries).astype(int)

        price_dev = (
            abs(df["Unit_Price_USD"] - df["Market_Spot_Price"])
            / df["Market_Spot_Price"]
        )
        w3_price = (price_dev > 0.08).astype(int)

        high_limit = df["Total_Value_USD"].quantile(0.99)
        w3_high    = (df["Total_Value_USD"] > high_limit).astype(int)

        w3_round   = (df["Total_Value_USD"] % 1000 == 0).astype(int)

        w3_pay     = (df["Payment_Method"] == "Open_Account").astype(int)

        small_limit = df["Total_Value_USD"].quantile(0.25)
        small_txn   = df["Total_Value_USD"] <= small_limit
        vendor_small = df.groupby("Vendor_ID")["Total_Value_USD"].transform(
            lambda x: (x <= small_limit).sum()
        )
        w3_smurf = (small_txn & (vendor_small >= 4)).astype(int)

        w3_score = w3_ofac + w3_price + w3_high + w3_round + w3_pay + w3_smurf
        w3_label = w3_score.apply(
            lambda s: "Suspicious" if s >= 2 else "Non-Suspicious"
        )

        # ── Overlap statistics ────────────────────────────────────────────
        w3_sus  = (w3_label == "Suspicious").sum()
        new_sus = (df["Suspicious_Label"] == "Suspicious").sum()

        both     = ((w3_label == "Suspicious") & (df["Suspicious_Label"] == "Suspicious")).sum()
        only_w3  = ((w3_label == "Suspicious") & (df["Suspicious_Label"] != "Suspicious")).sum()
        only_new = ((w3_label != "Suspicious") & (df["Suspicious_Label"] == "Suspicious")).sum()

        print(f"  Week 3 baseline flagged    :  {w3_sus:,} transactions")
        print(f"  Combined model flagged     :  {new_sus:,} transactions")
        print(f"  Flagged by both            :  {both:,}")
        print(f"  Only Week 3 (potential FP) :  {only_w3:,}")
        print(f"  Only combined (new catches):  {only_new:,}")
        if w3_sus > 0:
            recall = both / w3_sus * 100
            print(f"  Recall of Week 3 flags     :  {recall:.1f} %")

    # ── save output ───────────────────────────────────────────────────────────
    def save(self, result: pd.DataFrame, output_path: str):
        """Save the fully labelled result to an Excel file."""
        output_cols = [
            # Original data
            "Transaction_ID", "Date", "Vendor_ID", "Vendor_Country",
            "Commodity", "Volume_MT", "Market_Spot_Price", "Unit_Price_USD",
            "Total_Value_USD", "Payment_Method",
            # Module 1 – OFAC
            "OFAC_Flag", "Jurisdiction_Risk",
            # Module 2 – Pricing
            "Predicted_Price", "Price_Residual", "Price_ZScore", "Price_Anomaly",
            # Module 3 – Benford
            "Benford_ChiSq", "Benford_PValue", "Benford_Anomaly",
            # Module 4 – Smurfing
            "Smurf_CV_Score", "Smurf_Burst_Score", "Smurf_Threshold_Score",
            "Smurf_Combined", "Smurf_Anomaly",
            # Module 5 – Velocity
            "Velocity_Score", "Velocity_Anomaly",
            # Module 6 – Statistical
            "Mahalanobis_Dist", "Mahalanobis_PValue", "Statistical_Outlier",
            # Fusion
            "Suspicion_Index", "Suspicious_Label",
        ]
        available = [c for c in output_cols if c in result.columns]
        result[available].to_excel(output_path, index=False)
        print(f"\n  Saved labelled output -> {output_path}")
        print(f"     Columns: {len(available)}  |  Rows: {len(result):,}")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    INPUT_FILE  = "metallurgical_ledgers.xlsx"
    OUTPUT_FILE = "model_labelled_ledgers.xlsx"

    t0 = time.time()

    print(f"Loading data from {INPUT_FILE} ...")
    df = pd.read_excel(INPUT_FILE)
    print(f"Loaded {len(df):,} transactions.\n")

    # ── Run the combined model ────────────────────────────────────────────
    model  = FinancialAnomalyModel(suspicious_threshold=35)
    result = model.run(df)

    # ── Compare with Week 3 baseline ──────────────────────────────────────
    model.compare_with_week3(result)

    # ── Save ──────────────────────────────────────────────────────────────
    model.save(result, OUTPUT_FILE)

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f} seconds")
