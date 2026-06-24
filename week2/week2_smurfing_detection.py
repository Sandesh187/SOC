# week2_smurfing_detection.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SmurfingDetector:
    """
    Detect smurfing/structuring patterns in trade transactions
    """
    
    def __init__(self, reporting_threshold=10000):
        self.threshold = reporting_threshold
        self.suspicious_range = (0.85 * reporting_threshold, 0.999 * reporting_threshold)
    
    def detect_threshold_avoidance(self, transactions_df):
        """
        Flag transactions just below reporting threshold
        """
        df = transactions_df.copy()
        
        # Flag transactions in suspicious range
        df['near_threshold'] = (
            (df['amount'] >= self.suspicious_range[0]) & 
            (df['amount'] <= self.suspicious_range[1])
        )
        
        # Flag exact round amounts (common smurfing pattern)
        df['round_amount'] = df['amount'].apply(
            lambda x: x % 100 == 0 or x % 500 == 0 or x % 1000 == 0
        )
        
        # Combined smurfing indicator
        df['smurfing_flag'] = df['near_threshold'] & df['round_amount']
        
        return df
    
    def detect_rapid_succession(self, transactions_df, time_window_hours=24):
        """
        Detect multiple transactions from same entity within short window
        """
        df = transactions_df.copy()
        df['datetime'] = pd.to_datetime(df['date'])
        df = df.sort_values(['entity_id', 'datetime'])
        
        # Calculate time difference between consecutive transactions
        df['time_diff_hours'] = df.groupby('entity_id')['datetime'].diff().dt.total_seconds() / 3600
        
        # Flag rapid succession
        df['rapid_succession'] = df['time_diff_hours'] <= time_window_hours
        
        # Count transactions per entity per window
        window_counts = df.groupby('entity_id').rolling(
            f'{time_window_hours}H', on='datetime'
        )['amount'].count().reset_index()
        
        return df
    
    def detect_structuring_pattern(self, transactions_df, entity_id, date_range_days=30):
        """
        Analyze if an entity is structuring to avoid reporting
        """
        entity_df = transactions_df[transactions_df['entity_id'] == entity_id].copy()
        entity_df['date'] = pd.to_datetime(entity_df['date'])
        
        # Filter to date range
        end_date = entity_df['date'].max()
        start_date = end_date - timedelta(days=date_range_days)
        entity_df = entity_df[entity_df['date'] >= start_date]
        
        total_amount = entity_df['amount'].sum()
        transaction_count = len(entity_df)
        avg_amount = entity_df['amount'].mean()
        
        # Smurfing indicators
        below_threshold_pct = (entity_df['amount'] < self.threshold).mean() * 100
        near_threshold_count = (
            (entity_df['amount'] >= self.suspicious_range[0]) & 
            (entity_df['amount'] < self.threshold)
        ).sum()
        
        # Risk assessment
        risk_score = 0
        flags = []
        
        if total_amount > self.threshold * 3 and transaction_count >= 5:
            risk_score += 30
            flags.append("Multiple transactions totaling >3x threshold")
        
        if near_threshold_count >= 3:
            risk_score += 40
            flags.append(f"{near_threshold_count} transactions near threshold")
        
        if below_threshold_pct > 80 and transaction_count > 5:
            risk_score += 20
            flags.append("Most transactions below threshold")
        
        if transaction_count >= 10 and avg_amount < self.threshold * 0.5:
            risk_score += 10
            flags.append("High frequency, low average amount")
        
        return {
            'entity_id': entity_id,
            'total_amount': total_amount,
            'transaction_count': transaction_count,
            'avg_amount': avg_amount,
            'below_threshold_pct': below_threshold_pct,
            'near_threshold_count': near_threshold_count,
            'risk_score': min(risk_score, 100),
            'risk_level': 'HIGH' if risk_score >= 60 else 'MEDIUM' if risk_score >= 30 else 'LOW',
            'flags': flags
        }
    
    def generate_sample_transactions(self, n=100):
        """
        Generate sample transaction data for testing
        """
        np.random.seed(42)
        
        entities = ['ENT001', 'ENT002', 'ENT003', 'ENT004', 'ENT005']
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        
        transactions = []
        
        # Normal transactions (legitimate)
        for _ in range(60):
            transactions.append({
                'transaction_id': f'TXN{len(transactions):03d}',
                'entity_id': np.random.choice(entities),
                'date': np.random.choice(dates),
                'amount': np.random.uniform(5000, 50000),
                'type': 'normal'
            })
        
        # Smurfing pattern (suspicious)
        smurfer_entity = 'ENT003'
        smurf_dates = pd.date_range('2023-01-15', periods=5, freq='D')
        for i, date in enumerate(smurf_dates):
            transactions.append({
                'transaction_id': f'TXN{len(transactions):03d}',
                'entity_id': smurfer_entity,
                'date': date,
                'amount': 9500 + (i * 50),  # Just below 10K, slightly varied
                'type': 'smurfing'
            })
        
        # Another smurfing pattern (rapid succession)
        rapid_date = datetime(2023, 1, 20)
        for i in range(4):
            transactions.append({
                'transaction_id': f'TXN{len(transactions):03d}',
                'entity_id': 'ENT005',
                'date': rapid_date + timedelta(hours=i*2),
                'amount': 9800,
                'type': 'smurfing'
            })
        
        return pd.DataFrame(transactions)


if __name__ == "__main__":
    print("=" * 60)
    print("SMURFING DETECTION - WEEK 2")
    print("=" * 60)
    
    detector = SmurfingDetector(reporting_threshold=10000)
    
    # Generate sample data
    print("\n--- Generating Sample Transactions ---")
    transactions = detector.generate_sample_transactions(n=100)
    print(f"Generated {len(transactions)} transactions")
    print(transactions.head(10))
    
    # Detect threshold avoidance
    print("\n--- Threshold Avoidance Analysis ---")
    flagged = detector.detect_threshold_avoidance(transactions)
    smurf_count = flagged['smurfing_flag'].sum()
    print(f"Flagged transactions: {smurf_count}")
    print(flagged[flagged['smurfing_flag']][['transaction_id', 'entity_id', 'amount', 'smurfing_flag']])
    
    # Analyze specific entities
    print("\n--- Entity Risk Assessment ---")
    for entity in transactions['entity_id'].unique():
        result = detector.detect_structuring_pattern(transactions, entity)
        print(f"\n{entity}:")
        print(f"  Risk: {result['risk_level']} (Score: {result['risk_score']})")
        print(f"  Total: ${result['total_amount']:,.0f} across {result['transaction_count']} transactions")
        print(f"  Flags: {result['flags']}")
    
    # Save
    flagged.to_csv('smurfing_analysis_week2.csv', index=False)
    print(f"\n💾 Saved to smurfing_analysis_week2.csv")
    
    print("\n" + "=" * 60)
    print("Week 2 Smurfing Detection complete!")
    print("=" * 60)