# Week 3 Report: OFAC Sanctions & Transaction Suspicion Criteria

This report outlines the suspicious activity criteria designed to evaluate and label the transaction records in `metallurgical_ledgers.xlsx`.

## 1. OFAC Sanctions Screening
The Office of Foreign Assets Control (OFAC) of the US Department of the Treasury publishes lists of sanctioned countries and individuals.
* **Sanctioned/High-Risk Countries**: Iran, North Korea, Russia, Syria, Cuba, as well as designated proxy countries (`Sanctioned_Proxy_Alpha`, `Sanctioned_Proxy_Beta`).
* **Implementation**: Any transaction involving a vendor located in these countries is immediately flagged for jurisdiction risk.

## 2. Suspicion Criteria (6 Key Indicators)
To screen transactions, six binary criteria columns were generated:

1. **`OFAC_Country_Risk`**: Flags vendors operating in sanctioned or high-risk jurisdictions.
2. **`Price_Deviation_Risk`**: Flags transactions where the invoice unit price differs from the market spot price by more than 8%. Under-invoicing (evading customs/tariffs) or over-invoicing (transferring money out of a country) are classic TBML markers.
3. **`High_Value_Risk`**: Flags outlier transactions with a value above the 99th percentile of all ledgers.
4. **`Round_Number_Risk`**: Flags transactions with values that are exact multiples of $1,000 (highly uncommon in natural, weight-based commodity invoicing).
5. **`Payment_Method_Risk`**: Flags Open Account payment methods (which lack bank documentary checks and present higher trade-based fraud risk compared to Letters of Credit or Wire transfers).
6. **`Smurfing_Risk`**: Flags vendors conducting a high volume of small-value transactions (below the 25th percentile of transaction value, with at least 4 occurrences).

## 3. Labeling Methodology & Results
* **Suspicion Score**: Sum of the 6 risk flags.
* **Suspicious Label**: A transaction is labeled `Suspicious` if its `Suspicion_Score` $\ge 2$, meaning at least two warning signs are present concurrently.
* **Results**:
  - The script [week3_labelling.py](file:///c:/Users/SANDESH/summer_of_code/week3/week3_labelling.py) successfully processed the entire ledger of **15,030 transactions**.
  - The final labeled file is saved at [labelled_metallurgical_ledgers.xlsx](file:///c:/Users/SANDESH/summer_of_code/week5/labelled_metallurgical_ledgers.xlsx).

---
### Citations
* U.S. Department of the Treasury. (2023). *Office of Foreign Assets Control (OFAC) Sanctions Program and Country Information*.
* Egmont Group of Financial Intelligence Units. (2018). *Fiancial Analysis of Trade-Based Money Laundering cases*.
