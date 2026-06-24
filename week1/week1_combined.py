# week1_pipeline.py
from week1_yahoo_finance import YahooMetalsData
from week1_un_comtrade import UNComtradeAPI
import pandas as pd
from datetime import datetime

class MetalsTradePipeline:
    def __init__(self, comtrade_key=None):
        self.yahoo = YahooMetalsData()
        self.comtrade = UNComtradeAPI(subscription_key=comtrade_key)
    
    def run_week1(self, metal='copper', year='2023'):
        print("=" * 60)
        print("WEEK 1: DATA INGESTION PIPELINE")
        print("=" * 60)
        
        # 1. Yahoo Finance
        print("\n[1/2] Yahoo Finance - Price Data")
        print("-" * 40)
        price_data = self.yahoo.fetch_historical(metal, period='2y')
        print(f"Records: {len(price_data)}")
        print(f"Date range: {price_data.index.min()} to {price_data.index.max()}")
        print(f"Price range: ${price_data['Close'].min():.2f} - ${price_data['Close'].max():.2f}")
        self.yahoo.save_to_csv(metal, f'{metal}_prices_week1.csv')
        
        # 2. UN Comtrade
        print("\n[2/2] UN Comtrade - Trade Data")
        print("-" * 40)
        
        china_data = self.comtrade.fetch_china_copper_imports(year=year)
        chile_data = self.comtrade.fetch_chile_copper_exports(year=year)
        
        trade_records = []
        if not china_data.empty:
            china_proc = self.comtrade.process_data(china_data)
            print(f"China imports: {len(china_proc)} records")
            trade_records.append(china_proc)
        
        if not chile_data.empty:
            chile_proc = self.comtrade.process_data(chile_data)
            print(f"Chile exports: {len(chile_proc)} records")
            trade_records.append(chile_proc)
        
        if trade_records:
            combined_trade = pd.concat(trade_records, ignore_index=True)
            self.comtrade.save_data(combined_trade, 'trade_data_week1.csv')
            print(f"\nTotal trade records: {len(combined_trade)}")
        else:
            print("No trade data retrieved")
            combined_trade = pd.DataFrame()
        
        # Generate report
        self.generate_report(price_data, combined_trade)
        
        return price_data, combined_trade
    
    def generate_report(self, price_data, trade_data):
        report = []
        report.append("=" * 60)
        report.append("WEEK 1 REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("=" * 60)
        report.append("")
        report.append("--- YAHOO FINANCE ---")
        report.append(f"Metal: Copper (HG=F)")
        report.append(f"Records: {len(price_data)}")
        report.append(f"Latest Close: ${price_data['Close'].iloc[-1]:.2f}")
        report.append(f"Volatility: {price_data['Close'].pct_change().std() * np.sqrt(252):.2%}")
        report.append("")
        report.append("--- UN COMTRADE ---")
        if not trade_data.empty:
            report.append(f"Total Records: {len(trade_data)}")
            if 'Value_USD' in trade_data.columns:
                report.append(f"Total Trade Value: ${trade_data['Value_USD'].sum():,.0f}")
            if 'Reporter' in trade_data.columns:
                report.append(f"Countries: {trade_data['Reporter'].nunique()}")
        else:
            report.append("No trade data available")
        report.append("")
        report.append("--- NEXT STEPS ---")
        report.append("1. Sign up for full Comtrade API key at comtradeplus.un.org")
        report.append("2. Week 2: World Bank API + Benford's Law")
        
        report_text = "\n".join(report)
        print("\n" + report_text)
        
        with open('week1_report.txt', 'w') as f:
            f.write(report_text)
        print("\nSaved to week1_report.txt")


if __name__ == "__main__":
    import numpy as np
    
    pipeline = MetalsTradePipeline()
    prices, trade = pipeline.run_week1(metal='copper', year='2023')