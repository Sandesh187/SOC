# week1_un_comtrade.py
import comtradeapicall
import pandas as pd
import numpy as np
import time
from datetime import datetime

class UNComtradeAPI:
    """
    UN Comtrade API wrapper using official comtradeapicall library
    """
    
    def __init__(self, subscription_key=None):
        self.subscription_key = subscription_key
        self.hs_codes = {
            'copper_refined': '7403',
            'copper_ores': '2603',
            'copper_scrap': '7404',
            'gold': '7108',
            'aluminum_unwrought': '7601',
            'aluminum_scrap': '7602'
        }
        self.country_codes = {
            'CHN': '156',  # China
            'USA': '842',  # United States
            'DEU': '276',  # Germany
            'JPN': '392',  # Japan
            'IND': '699',  # India
            'GBR': '826',  # UK
            'RUS': '643',  # Russia
            'BRA': '076',  # Brazil
            'ZAF': '710',  # South Africa
            'AUS': '036',  # Australia
            'CHL': '152',  # Chile (major copper producer)
            'PER': '604',  # Peru (major copper producer)
        }
    
    def fetch_preview(self, commodity='copper_refined', reporter='156', 
                      partner=None, year='2023', trade_regime='M'):
        """
        Fetch data WITHOUT API key (max 500 records)
        """
        hs_code = self.hs_codes.get(commodity, commodity)
        
        print(f"Fetching {commodity} (HS:{hs_code}) for {year}, Reporter:{reporter}...")
        
        try:
            df = comtradeapicall.previewFinalData(
                typeCode='C',
                freqCode='A',
                clCode='HS',
                period=year,
                reporterCode=reporter,
                cmdCode=hs_code,
                flowCode=trade_regime,
                partnerCode=partner,
                partner2Code=None,
                customsCode=None,
                motCode=None,
                maxRecords=500,
                format_output='JSON',
                aggregateBy=None,
                breakdownMode='classic',
                countOnly=None,
                includeDesc=True
            )
            
            print(f"Fetched {len(df)} records")
            return df
            
        except Exception as e:
            print(f"Error: {e}")
            return pd.DataFrame()
    
    def fetch_full(self, commodity='copper_refined', reporter='156', 
                   partner=None, year='2023', trade_regime='M',
                   max_records=25000):
        """
        Fetch data WITH API key (max 250,000 records)
        """
        if not self.subscription_key:
            print("No API key provided! Use fetch_preview() or add your key.")
            return pd.DataFrame()
        
        hs_code = self.hs_codes.get(commodity, commodity)
        
        print(f"Fetching {commodity} (HS:{hs_code}) for {year}...")
        
        try:
            df = comtradeapicall.getFinalData(
                subscription_key=self.subscription_key,
                typeCode='C',
                freqCode='A',
                clCode='HS',
                period=year,
                reporterCode=reporter,
                cmdCode=hs_code,
                flowCode=trade_regime,
                partnerCode=partner,
                partner2Code=None,
                customsCode=None,
                motCode=None,
                maxRecords=max_records,
                format_output='JSON',
                aggregateBy=None,
                breakdownMode='classic',
                countOnly=None,
                includeDesc=True
            )
            
            print(f"Fetched {len(df)} records")
            return df
            
        except Exception as e:
            print(f"Error: {e}")
            return pd.DataFrame()
    
    def fetch_china_copper_imports(self, year='2023', use_preview=True):
        reporter_code = self.country_codes['CHN']
        if use_preview or not self.subscription_key:
            return self.fetch_preview(
                commodity='copper_refined',
                reporter=reporter_code,
                year=year,
                trade_regime='M'
            )
        else:
            return self.fetch_full(
                commodity='copper_refined',
                reporter=reporter_code,
                year=year,
                trade_regime='M'
            )
    
    def fetch_chile_copper_exports(self, year='2023', use_preview=True):
        reporter_code = self.country_codes['CHL']
        if use_preview or not self.subscription_key:
            return self.fetch_preview(
                commodity='copper_refined',
                reporter=reporter_code,
                year=year,
                trade_regime='X'
            )
        else:
            return self.fetch_full(
                commodity='copper_refined',
                reporter=reporter_code,
                year=year,
                trade_regime='X'
            )
    
    def process_data(self, df):
        if df.empty:
            return df
        
        column_map = {
            'reporterDesc': 'Reporter',
            'partnerDesc': 'Partner',
            'flowDesc': 'Trade_Flow',
            'cmdCode': 'HS_Code',
            'cmdDesc': 'Product',
            'period': 'Year',
            'primaryValue': 'Value_USD',
            'netWgt': 'Quantity_kg',
            'qtyUnitAbbr': 'Unit',
            'fobvalue': 'FOB_Value',
            'cifvalue': 'CIF_Value',
        }
        
        available_cols = {k: v for k, v in column_map.items() if k in df.columns}
        if not available_cols:
            return df
        
        processed = df[list(available_cols.keys())].copy()
        processed = processed.rename(columns=available_cols)
        
        numeric_cols = ['Value_USD', 'Quantity_kg', 'FOB_Value', 'CIF_Value']
        for col in numeric_cols:
            if col in processed.columns:
                processed[col] = pd.to_numeric(processed[col], errors='coerce')
        
        if 'Value_USD' in processed.columns and 'Quantity_kg' in processed.columns:
            processed['Unit_Price'] = processed['Value_USD'] / processed['Quantity_kg']
            processed['Unit_Price'] = processed['Unit_Price'].replace([np.inf, -np.inf], np.nan)
        
        return processed
    
    def save_data(self, df, filename=None):
        if df.empty:
            return None
        
        if filename is None:
            filename = f"comtrade_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        
        df.to_csv(filename, index=False)
        print(f"Saved to {filename}")
        return filename


if __name__ == "__main__":
    print("=" * 60)
    print("UN COMTRADE API - WEEK 1")
    print("=" * 60)
    
    comtrade = UNComtradeAPI(subscription_key=None)
    
    print("\n--- China Copper Imports (2023) ---")
    china_data = comtrade.fetch_china_copper_imports(year='2023', use_preview=True)
    
    if not china_data.empty:
        processed = comtrade.process_data(china_data)
        print(f"\nPreview data ({len(processed)} records):")
        print(processed.head())
        if 'Value_USD' in processed.columns:
            print(f"\nTotal value: ${processed['Value_USD'].sum():,.0f}")
            print(f"Top partners by value:")
            print(processed.groupby('Partner')['Value_USD'].sum().sort_values(ascending=False).head())
        comtrade.save_data(processed, 'china_copper_imports_2023.csv')
    
    time.sleep(1)
    
    print("\n--- Chile Copper Exports (2023) ---")
    chile_data = comtrade.fetch_chile_copper_exports(year='2023', use_preview=True)
    
    if not chile_data.empty:
        processed_chile = comtrade.process_data(chile_data)
        print(f"\nPreview data ({len(processed_chile)} records):")
        print(processed_chile.head())
        comtrade.save_data(processed_chile, 'chile_copper_exports_2023.csv')
    
    print("\n" + "=" * 60)
    print("Week 1 Comtrade tasks complete!")
    print("=" * 60)