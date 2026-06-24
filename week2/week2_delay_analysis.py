# week2_delay_analysis.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ShipmentDelayAnalyzer:
    """
    Analyze deal-to-shipment delays for compliance risk
    """
    
    def __init__(self):
        # Standard shipping routes (days) for copper trade
        self.standard_routes = {
            ('CHL', 'CHN'): 45,   # Chile to China (major route)
            ('PER', 'CHN'): 50,   # Peru to China
            ('AUS', 'CHN'): 35,   # Australia to China
            ('CHL', 'USA'): 30,   # Chile to US (Panama Canal)
            ('PER', 'USA'): 35,   # Peru to US
            ('ZAF', 'CHN'): 55,   # South Africa to China
            ('RUS', 'CHN'): 40,   # Russia to China (land/sea mix)
            ('CHL', 'DEU'): 55,   # Chile to Germany
            ('AUS', 'JPN'): 25,   # Australia to Japan
            ('CAN', 'USA'): 15,   # Canada to US (land)
            ('CHL', 'IND'): 50,   # Chile to India
        }
        
        # Incoterms and their typical delay modifiers
        self.incoterms = {
            'FOB': 0,      # Baseline
            'CIF': 5,      # +5 days for insurance docs
            'CFR': 3,      # +3 days for freight docs
            'DAP': 10,     # +10 days for delivery
            'EXW': 15,     # +15 days for buyer arrangement
            'DDP': 12,     # +12 days for full delivery
        }
    
    def calculate_expected_delay(self, origin, destination, incoterm='FOB'):
        """
        Calculate expected shipping delay based on route and Incoterm
        """
        route = (origin, destination)
        
        # Base delay from route database
        if route in self.standard_routes:
            base_delay = self.standard_routes[route]
        else:
            # Estimate based on hemisphere and distance
            base_delay = self._estimate_delay(origin, destination)
        
        # Add Incoterm modifier
        incoterm_modifier = self.incoterms.get(incoterm.upper(), 0)
        
        expected_delay = base_delay + incoterm_modifier
        
        return {
            'origin': origin,
            'destination': destination,
            'incoterm': incoterm,
            'base_delay_days': base_delay,
            'incoterm_modifier': incoterm_modifier,
            'expected_delay_days': expected_delay,
            'normal_range': (expected_delay - 10, expected_delay + 15)
        }
    
    def _estimate_delay(self, origin, destination):
        """
        Rough estimation for unknown routes
        """
        # Simplified estimation - in real system use distance calculation
        return 45  # Default 45 days
    
    def assess_delay_risk(self, deal_date, shipment_date, origin, destination, incoterm='FOB'):
        """
        Assess if actual delay is anomalous
        """
        expected = self.calculate_expected_delay(origin, destination, incoterm)
        actual_delay = (shipment_date - deal_date).days
        
        expected_delay = expected['expected_delay_days']
        normal_min, normal_max = expected['normal_range']
        
        # Risk classification
        if actual_delay < 0:
            risk_level = 'INVALID'
            risk_score = 100
            flag = 'Shipment before deal date - impossible'
        elif actual_delay < normal_min:
            risk_level = 'LOW'
            risk_score = 10
            flag = 'Fast shipping - verify air freight or nearby origin'
        elif normal_min <= actual_delay <= normal_max:
            risk_level = 'NORMAL'
            risk_score = 0
            flag = 'Within normal range'
        elif normal_max < actual_delay <= normal_max + 30:
            risk_level = 'MEDIUM'
            risk_score = 50
            flag = 'Slightly delayed - monitor for patterns'
        elif normal_max + 30 < actual_delay <= normal_max + 90:
            risk_level = 'HIGH'
            risk_score = 80
            flag = 'Significant delay - possible rerouting or storage (TBML indicator)'
        else:
            risk_level = 'CRITICAL'
            risk_score = 100
            flag = 'Extreme delay - strong fraud indicator'
        
        return {
            'deal_date': deal_date,
            'shipment_date': shipment_date,
            'actual_delay_days': actual_delay,
            'expected_delay_days': expected_delay,
            'delay_deviation': actual_delay - expected_delay,
            'risk_level': risk_level,
            'risk_score': risk_score,
            'flag': flag,
            'is_anomaly': risk_score >= 50
        }
    
    def generate_sample_analysis(self):
        """
        Generate sample analysis for report
        """
        print("=" * 60)
        print("DEAL-SHIPMENT DELAY ANALYSIS")
        print("=" * 60)
        
        test_cases = [
            ('CHL', 'CHN', 'FOB', 45),      # Normal
            ('CHL', 'CHN', 'FOB', 120),     # High risk - TBML?
            ('CHL', 'CHN', 'FOB', 200),     # Critical - fraud
            ('AUS', 'JPN', 'CIF', 30),      # Normal
            ('PER', 'USA', 'FOB', 35),      # Normal
            ('RUS', 'CHN', 'FOB', 95),      # High - sanctions evasion?
        ]
        
        results = []
        deal_date = datetime(2023, 1, 15)
        
        for origin, dest, incoterm, actual_days in test_cases:
            shipment = deal_date + timedelta(days=actual_days)
            result = self.assess_delay_risk(deal_date, shipment, origin, dest, incoterm)
            results.append(result)
            
            print(f"\nRoute: {origin} → {dest} ({incoterm})")
            print(f"  Expected: {result['expected_delay_days']} days")
            print(f"  Actual: {result['actual_delay_days']} days")
            print(f"  Risk: {result['risk_level']} (Score: {result['risk_score']})")
            print(f"  Flag: {result['flag']}")
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    analyzer = ShipmentDelayAnalyzer()
    df = analyzer.generate_sample_analysis()
    
    # Save to CSV
    df.to_csv('delay_analysis_week2.csv', index=False)
    print(f"\n💾 Saved to delay_analysis_week2.csv")
    
    print("\n" + "=" * 60)
    print("Week 2 Delay Analysis complete!")
    print("=" * 60)