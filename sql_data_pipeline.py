#!/usr/bin/env python3
"""
sql_data_pipeline.py — SQL & Python Large Data Handling for TBML Detection
==========================================================================
Demonstrates the integration of SQL (via SQLite / Python sqlite3) to store,
query, index, and aggregate large trade ledger datasets (15,000+ transactions)
for Trade-Based Money Laundering (TBML) compliance monitoring.

Features:
  1. Automated DB Schema setup with indexed fields (Transaction_ID, Vendor_ID, Date, Country)
  2. Bulk Data Ingestion from Excel to SQLite DB
  3. SQL Window Functions for High-Velocity Transaction Detection
  4. SQL Structuring / Smurfing Queries (transactions near reporting thresholds)
  5. SQL Sanctions & High-Risk Jurisdiction Filtering
  6. SQL Price Deviation Aggregations (Over/Under-invoicing vs. Market Spot Price)
  7. Consolidated SQL Risk View (`suspicious_transactions_view`) for compliance reporting
"""

import sqlite3
import pandas as pd
import numpy as np
import os
import time

DB_FILE = "trade_compliance.db"
EXCEL_FILE = "metallurgical_ledgers.xlsx"

def setup_sql_database(excel_path=EXCEL_FILE, db_path=DB_FILE):
    """
    Ingests Excel dataset into an indexed SQLite relational database.
    """
    print("=" * 70)
    print(" 1. SQL DATABASE SETUP & INGESTION")
    print("=" * 70)
    
    start_time = time.time()
    df = pd.read_excel(excel_path)
    print(f"[*] Read {len(df):,} rows from '{excel_path}'")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop table if exists for clean run
    cursor.execute("DROP TABLE IF EXISTS transactions")
    cursor.execute("DROP VIEW IF EXISTS suspicious_transactions_view")
    
    # Create optimized table schema with SQL constraints
    create_table_sql = """
    CREATE TABLE transactions (
        Transaction_ID TEXT PRIMARY KEY,
        Date TEXT NOT NULL,
        Vendor_ID TEXT NOT NULL,
        Vendor_Country TEXT NOT NULL,
        Commodity TEXT NOT NULL,
        Volume_MT REAL NOT NULL,
        Market_Spot_Price REAL NOT NULL,
        Unit_Price_USD REAL NOT NULL,
        Total_Value_USD REAL NOT NULL,
        Payment_Method TEXT NOT NULL
    );
    """
    cursor.execute(create_table_sql)
    
    # Convert Date to string ISO format for SQL sorting
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    
    # Bulk insert into SQL table
    df.to_sql('transactions', conn, if_exists='append', index=False)
    
    # Create SQL Indexes for high-speed queries on large data
    cursor.execute("CREATE INDEX idx_vendor ON transactions(Vendor_ID);")
    cursor.execute("CREATE INDEX idx_date ON transactions(Date);")
    cursor.execute("CREATE INDEX idx_country ON transactions(Vendor_Country);")
    cursor.execute("CREATE INDEX idx_commodity ON transactions(Commodity);")
    
    conn.commit()
    elapsed = time.time() - start_time
    print(f"[+] Loaded {len(df):,} records into SQLite table 'transactions' in {elapsed:.3f}s")
    print(f"[+] Created 4 SQL indexes for optimized large-data querying.")
    conn.close()


def run_sql_tbml_analytics(db_path=DB_FILE):
    """
    Executes complex SQL queries to identify TBML indicators.
    """
    print("\n" + "=" * 70)
    print(" 2. EXECUTING SQL QUERIES FOR TBML MONITORING")
    print("=" * 70)
    
    conn = sqlite3.connect(db_path)
    
    # -------------------------------------------------------------
    # SQL QUERY 1: Sanctioned & Proxy Country Screening
    # -------------------------------------------------------------
    print("\n---> [SQL Query 1] Sanctioned Jurisdiction Screening")
    sql_sanctions = """
    SELECT 
        Vendor_Country,
        COUNT(*) AS Total_Transactions,
        ROUND(SUM(Total_Value_USD), 2) AS Total_Volume_USD,
        ROUND(AVG(Total_Value_USD), 2) AS Avg_Txn_Value
    FROM transactions
    WHERE Vendor_Country LIKE '%Sanctioned%' OR Vendor_Country IN ('Iran', 'North Korea', 'Syria', 'Russia')
    GROUP BY Vendor_Country
    ORDER BY Total_Volume_USD DESC;
    """
    df_sanctions = pd.read_sql_query(sql_sanctions, conn)
    print(df_sanctions.to_string(index=False))

    # -------------------------------------------------------------
    # SQL QUERY 2: Smurfing / Structuring Detection ($8,000 - $9,999)
    # -------------------------------------------------------------
    print("\n---> [SQL Query 2] Smurfing/Structuring Pattern Query (Txns near $10k limit)")
    sql_smurfing = """
    SELECT 
        Vendor_ID,
        Vendor_Country,
        COUNT(*) AS Structured_Txn_Count,
        ROUND(SUM(Total_Value_USD), 2) AS Total_Structured_USD,
        MIN(Date) AS First_Txn_Date,
        MAX(Date) AS Last_Txn_Date
    FROM transactions
    WHERE Total_Value_USD BETWEEN 8000 AND 9999
    GROUP BY Vendor_ID, Vendor_Country
    HAVING Structured_Txn_Count >= 3
    ORDER BY Structured_Txn_Count DESC
    LIMIT 10;
    """
    df_smurfing = pd.read_sql_query(sql_smurfing, conn)
    print(df_smurfing.to_string(index=False))

    # -------------------------------------------------------------
    # SQL QUERY 3: Price Deviation (Over/Under-Invoicing vs. Spot Price)
    # -------------------------------------------------------------
    print("\n---> [SQL Query 3] Over/Under-Invoicing SQL Aggregations by Commodity")
    sql_price_dev = """
    SELECT 
        Commodity,
        COUNT(*) AS Txn_Count,
        ROUND(AVG(Unit_Price_USD), 2) AS Avg_Unit_Price,
        ROUND(AVG(Market_Spot_Price), 2) AS Avg_Spot_Price,
        ROUND(AVG(ABS(Unit_Price_USD - Market_Spot_Price) / Market_Spot_Price) * 100, 2) AS Avg_Price_Dev_Pct,
        COUNT(CASE WHEN ABS(Unit_Price_USD - Market_Spot_Price) / Market_Spot_Price > 0.15 THEN 1 END) AS High_Deviation_Count
    FROM transactions
    GROUP BY Commodity
    ORDER BY High_Deviation_Count DESC;
    """
    df_price = pd.read_sql_query(sql_price_dev, conn)
    print(df_price.to_string(index=False))

    # -------------------------------------------------------------
    # SQL QUERY 4: Transaction Velocity via SQL Window Functions
    # -------------------------------------------------------------
    print("\n---> [SQL Query 4] Transaction Velocity Spikes using SQL Window Functions")
    sql_velocity = """
    WITH Velocity_Calc AS (
        SELECT 
            Transaction_ID,
            Date,
            Vendor_ID,
            Vendor_Country,
            Total_Value_USD,
            COUNT(*) OVER (
                PARTITION BY Vendor_ID 
                ORDER BY Date 
                RANGE BETWEEN 7 PRECEDING AND CURRENT ROW
            ) AS Txns_Last_7_Days
        FROM transactions
    )
    SELECT 
        Vendor_ID,
        Vendor_Country,
        COUNT(*) AS High_Velocity_Events,
        MAX(Txns_Last_7_Days) AS Max_Txns_In_7_Days
    FROM Velocity_Calc
    WHERE Txns_Last_7_Days >= 10
    GROUP BY Vendor_ID, Vendor_Country
    ORDER BY High_Velocity_Events DESC
    LIMIT 10;
    """
    df_velocity = pd.read_sql_query(sql_velocity, conn)
    print(df_velocity.to_string(index=False))

    # -------------------------------------------------------------
    # SQL QUERY 5: Creating Consolidated SQL Compliance View
    # -------------------------------------------------------------
    print("\n---> [SQL Query 5] Creating Consolidated SQL View: `suspicious_transactions_view`")
    sql_create_view = """
    CREATE VIEW IF NOT EXISTS suspicious_transactions_view AS
    SELECT 
        t.Transaction_ID,
        t.Date,
        t.Vendor_ID,
        t.Vendor_Country,
        t.Commodity,
        t.Total_Value_USD,
        t.Payment_Method,
        CASE WHEN t.Vendor_Country LIKE '%Sanctioned%' THEN 1 ELSE 0 END AS Flag_Sanction,
        CASE WHEN t.Total_Value_USD BETWEEN 8000 AND 9999 THEN 1 ELSE 0 END AS Flag_Smurfing,
        CASE WHEN ABS(t.Unit_Price_USD - t.Market_Spot_Price) / t.Market_Spot_Price > 0.15 THEN 1 ELSE 0 END AS Flag_Price_Dev
    FROM transactions t;
    """
    conn.cursor().execute(sql_create_view)
    conn.commit()

    # Query the view
    sql_view_summary = """
    SELECT 
        COUNT(*) AS Total_Flags_In_View,
        SUM(Flag_Sanction) AS Sanction_Flags,
        SUM(Flag_Smurfing) AS Smurfing_Flags,
        SUM(Flag_Price_Dev) AS Price_Dev_Flags
    FROM suspicious_transactions_view;
    """
    df_view = pd.read_sql_query(sql_view_summary, conn)
    print(df_view.to_string(index=False))
    
    conn.close()
    print("\n[+] SQL Data Handling & Query execution completed successfully.")

if __name__ == "__main__":
    setup_sql_database()
    run_sql_tbml_analytics()
