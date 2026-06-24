# week2_world_bank.py
import wbgapi as wb
import pandas as pd
import numpy as np
from datetime import datetime

class WorldBankAnalyzer:
    """
    Fetch macroeconomic indicators for metals trading countries
    """
    
    def __init__(self):
        # Key indicators for risk assessment
        self.indicators = {
            'NY.GDP.PCAP.CD': 'GDP_per_capita',
            'NE.TRD.GNFS.ZS': 'Trade_openness_pct_GDP',
            'CC.EST': 'Control_of_Corruption',
            'RL.EST': 'Rule_of_Law',
            'RQ.EST': 'Regulatory_Quality',
            'GE.EST': 'Government_Effectiveness',
            'VA.EST': 'Voice_Accountability',
            'PV.EST': 'Political_Stability',
        }
        
        # Major metals trading countries (ISO3 codes)
        self.countries = [
            'CHN', 'USA', 'DEU', 'JPN', 'IND',  # Major economies
            'CHL', 'PER', 'AUS', 'ZAF', 'RUS',  # Major producers
            'BRA', 'CAN', 'MEX', 'KOR', 'IDN',  # Significant traders
            'IRN', 'ARE', 'SAU', 'TUR', 'POL',  # Additional markets
        ]
    
    def fetch_indicators(self, years=range(2018, 2024)):
        """
        Fetch all indicators for all countries
        """
        print(f"Fetching World Bank data for {len(self.countries)} countries...")
        print(f"Indicators: {list(self.indicators.values())}")
        
        try:
            data = wb.data.DataFrame(
                list(self.indicators.keys()),
                self.countries,
                time=years,
                labels=False,
                numericTimeKeys=True,
                columns='series'
            )
            
            # Rename columns
            data = data.rename(columns=self.indicators)
            
            print(f"✅ Fetched {len(data)} records")
            return data
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return pd.DataFrame()
    
    def fetch_single_indicator(self, indicator_code, country='CHN', years=range(2020, 2024)):
        """
        Fetch single indicator for deep analysis
        """
        try:
            data = wb.data.DataFrame(
                indicator_code,
                country,
                time=years,
                numericTimeKeys=True
            )
            return data
        except Exception as e:
            print(f"Error fetching {indicator_code}: {e}")
            return pd.DataFrame()
    
    def calculate_risk_scores(self, df):
        """
        Calculate composite risk scores from indicators
        Higher score = higher risk
        """
        if df.empty:
            return df
        
        # Normalize corruption/control indicators (range -2.5 to 2.5)
        # Lower values = worse governance = higher risk
        risk_df = df.copy()
        
        # Invert so higher = more risk
        governance_cols = ['Control_of_Corruption', 'Rule_of_Law', 
                          'Regulatory_Quality', 'Government_Effectiveness']
        
        for col in governance_cols:
            if col in risk_df.columns:
                risk_df[f'{col}_risk'] = -risk_df[col]  # Invert: low score = high risk
        
        # Trade openness: very low or very high can indicate risk
        if 'Trade_openness_pct_GDP' in risk_df.columns:
            trade_median = risk_df['Trade_openness_pct_GDP'].median()
            risk_df['Trade_concentration_risk'] = abs(
                risk_df['Trade_openness_pct_GDP'] - trade_median
            ) / trade_median
        
        # Composite risk score (simple average of normalized risks)
        risk_cols = [c for c in risk_df.columns if '_risk' in c]
        if risk_cols:
            risk_df['Composite_Risk_Score'] = risk_df[risk_cols].mean(axis=1)
        
        return risk_df
    
    def save_data(self, df, filename='world_bank_indicators.csv'):
        df.to_csv(filename)
        print(f"💾 Saved to {filename}")
        return filename


if __name__ == "__main__":
    print("=" * 60)
    print("WORLD BANK API - WEEK 2")
    print("=" * 60)
    
    wb_analyzer = WorldBankAnalyzer()
    
    # Fetch all indicators
    print("\n--- Fetching Macroeconomic Indicators ---")
    indicators = wb_analyzer.fetch_indicators(years=range(2020, 2024))
    
    if not indicators.empty:
        print(f"\nData shape: {indicators.shape}")
        print(f"\nSample (China 2023):")
        print(indicators.loc['CHN'] if 'CHN' in indicators.index else indicators.head())
        
        # Calculate risk scores
        print("\n--- Calculating Risk Scores ---")
        risk_data = wb_analyzer.calculate_risk_scores(indicators)
        print(risk_data[['Composite_Risk_Score']].dropna().head(10))
        
        # Save
        wb_analyzer.save_data(indicators, 'world_bank_data_week2.csv')
        wb_analyzer.save_data(risk_data, 'world_bank_risk_scores_week2.csv')
    
    print("\n" + "=" * 60)
    print("Week 2 World Bank tasks complete!")
    print("=" * 60)