# Week 1 Report: Data Ingestion (Yahoo Finance & UN Comtrade)

This report summarizes the research and implementation of the data ingestion pipeline using the Yahoo Finance and UN Comtrade APIs.

## 1. Yahoo Finance API
Yahoo Finance is used to ingest historical market price data for commodities (specifically copper).
* **Ticker Used**: `HG=F` (Copper Futures on COMEX)
* **Access Method**: Python library `yfinance`.
* **Data Fields**: Open, High, Low, Close, Adj Close, Volume.
* **Findings**: Ingested historical daily spot prices for analysis. This data provides the baseline market spot price (`Market_Spot_Price`) against which transactional unit prices are evaluated for deviation.

## 2. UN Comtrade API
The United Nations Commodity Trade Statistics Database (UN Comtrade) provides detailed global trade flow data.
* **Commodity Code (HS)**: `7403` (Refined copper and copper alloys, unwrought).
* **API Details**: Ingested trade data using the official `comtradeapicall` Python library.
* **Data Flows**: 
  - **China Imports** (`flowCode='M'`): Tracking major buyer inflows.
  - **Chile Exports** (`flowCode='X'`): Tracking major seller outflows.
* **Data Fields**: Reporter, Partner, Trade Flow, Value (USD), Net Weight (kg), and calculated Unit Price ($/kg).

## 3. Combined Pipeline
The script [week1_combined.py](file:///c:/Users/SANDESH/summer_of_code/week1/week1_combined.py) integrates both sources to enable:
* Spot price trend tracking.
* Cross-border volume/value flow comparisons.
* Initial pricing deviation checks (comparing trade invoice unit prices vs. daily spot market prices).

---
### Citations
* United Nations Statistics Division. (2023). *UN Comtrade Database*. Retrieved from [https://comtradeplus.un.org/](https://comtradeplus.un.org/).
* Yahoo Finance. (2023). *Copper Futures (HG=F) Historical Data*. Retrieved from [https://finance.yahoo.com/](https://finance.yahoo.com/).
