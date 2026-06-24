# Week 2 Report: Macro Indicators, Shipment Delays, Smurfing, & Benford's Law

This report covers the research, indicators, and detection algorithms implemented for Week 2.

## 1. World Bank API & Country Risk
Using the `wbgapi` library, we ingested key governance and macroeconomic indicators for 20 major metals-trading countries.
* **Key Indicators**: 
  - `NY.GDP.PCAP.CD` (GDP per capita)
  - `NE.TRD.GNFS.ZS` (Trade openness)
  - Governance indicators (`CC.EST` for Control of Corruption, `RL.EST` for Rule of Law, `RQ.EST` for Regulatory Quality, `GE.EST` for Government Effectiveness).
* **Composite Risk Score**: Calculated by inverting and normalizing governance indicators. Lower scores indicate weak institutional controls, which elevate transaction risk (e.g., higher bribery/sanctions evasion risk).

## 2. Deal-to-Shipment Delays
Shipment delays can indicate Trade-Based Money Laundering (TBML), phantom shipments, or illicit rerouting to evade sanctions.
* **Baseline Assumptions**: Derived standard oceanic transit times (e.g., Chile to China: 45 days; Russia to China: 40 days; Peru to USA: 35 days).
* **Incoterm Modifiers**: Adjustments made for documentation overhead (e.g., CIF adds 5 days for insurance; EXW adds 15 days for buyer arrangements).
* **Risk Flags**: Delays exceeding the expected range by more than 30 days are flagged as `HIGH` risk, and those over 90 days are flagged as `CRITICAL` risk (potential storage/rerouting fraud).

## 3. Smurfing (Structuring)
Smurfing is a method where a single large transaction is broken down into multiple smaller transfers to remain under regulatory reporting thresholds (typically $10,000).
* **Detection Logic**: 
  - **Threshold Avoidance**: Flagging transaction amounts in the suspicious range of $8,500–$9,999.
  - **Rapid Succession**: Monitoring entities that perform multiple transactions within a 24-hour window.
  - **Frequency Checks**: Identifying vendors with a high ratio of low-value transactions compared to their historic average.

## 4. Benford's Law
Benford's Law states that the leading digits of naturally occurring numerical datasets follow a specific logarithmic distribution:
\[P(d) = \log_{10}\left(1 + \frac{1}{d}\right) \quad \text{for } d \in \{1, 2, \dots, 9\}\]
* **Application**: Real transactional values (like weights, invoice amounts) conform to this distribution. When fraudsters fabricate numbers, they tend to over-use digits like 5, 7, or 9, causing significant statistical deviations from the Benford profile.
* **Forensic Utility**: Used by auditors as an initial screening tool to highlight clusters of anomalous transactions for deep-dive reviews.

---
### Citations
* World Bank. (2023). *Worldwide Governance Indicators (WGI)*. Retrieved from [https://databank.worldbank.org/](https://databank.worldbank.org/).
* Financial Action Task Force (FATF). (2020). *Trade-Based Money Laundering: Trends and Developments*. FATF Report.
* Benford, F. (1938). *The Law of Anomalous Numbers*. Proceedings of the American Philosophical Society, 78(4), 551-572.
* Nigrini, M. J. (2012). *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection*. John Wiley & Sons.
