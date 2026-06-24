import pandas as pd

# -----------------------------
# 1. Load the Excel file
# -----------------------------
input_file = "metallurgical_ledgers.xlsx"
output_file = "week3/labelled_metallurgical_ledgers.xlsx"

df = pd.read_excel(input_file)


# -----------------------------
# 2. Create suspicious criteria
# -----------------------------

# Criterion 1: Vendor country is sanctioned / high risk
sanctioned_countries = [
    "Sanctioned_Proxy_Alpha",
    "Sanctioned_Proxy_Beta",
    "Iran",
    "North Korea",
    "Russia",
    "Syria",
    "Cuba",
]

df["OFAC_Country_Risk"] = df["Vendor_Country"].isin(sanctioned_countries).astype(int)


# Criterion 2: Unit price is more than 8% different from market spot price
price_difference = abs(df["Unit_Price_USD"] - df["Market_Spot_Price"]) / df["Market_Spot_Price"]
df["Price_Deviation_Risk"] = (price_difference > 0.08).astype(int)


# Criterion 3: Very high value transaction compared to other transactions
high_value_limit = df["Total_Value_USD"].quantile(0.99)
df["High_Value_Risk"] = (df["Total_Value_USD"] > high_value_limit).astype(int)


# Criterion 4: Round number transaction value, which can look artificial
round_value = df["Total_Value_USD"] % 1000 == 0
df["Round_Number_Risk"] = round_value.astype(int)


# Criterion 5: Open account payment method is riskier than wire / letter of credit
open_account = df["Payment_Method"] == "Open_Account"
df["Payment_Method_Risk"] = open_account.astype(int)


# Criterion 6: Possible smurfing - same vendor making many small transactions
small_value_limit = df["Total_Value_USD"].quantile(0.25)
df["Small_Transaction"] = df["Total_Value_USD"] <= small_value_limit

vendor_small_count = df.groupby("Vendor_ID")["Small_Transaction"].transform("sum")
df["Smurfing_Risk"] = ((df["Small_Transaction"] == True) & (vendor_small_count >= 4)).astype(int)


# -----------------------------
# 3. Final suspicious label
# -----------------------------
risk_columns = [
    "OFAC_Country_Risk",
    "Price_Deviation_Risk",
    "High_Value_Risk",
    "Round_Number_Risk",
    "Payment_Method_Risk",
    "Smurfing_Risk",
]

df["Suspicion_Score"] = df[risk_columns].sum(axis=1)

df["Suspicious_Label"] = df["Suspicion_Score"].apply(
    lambda score: "Suspicious" if score >= 2 else "Non-Suspicious"
)


# -----------------------------
# 4. Save the labelled Excel file
# -----------------------------
df.to_excel(output_file, index=False)

print("Excel file loaded from:", input_file)
print("Labelled file saved to:", output_file)
print("Total transactions:", len(df))
print("Suspicious transactions:", (df["Suspicious_Label"] == "Suspicious").sum())
