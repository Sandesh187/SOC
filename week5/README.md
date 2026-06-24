# Week 5 Report

## Task status: Week 3 labelling completed

The Week 3 transaction labelling task has been completed. The Excel file `metallurgical_ledgers.xlsx` was labelled using rule-based suspicious-transaction criteria. The code is in `week3/week3_labelling.py`, and a Week 5 copy of the labelled workbook is saved as `week5/labelled_metallurgical_ledgers.xlsx`.

The criteria used were:

- `OFAC_Country_Risk`: vendor country appears in the sanctioned or high-risk list.
- `Price_Deviation_Risk`: unit price differs from market spot price by more than 8%.
- `High_Value_Risk`: transaction value is above the 99th percentile.
- `Round_Number_Risk`: transaction value is an exact multiple of 1,000.
- `Payment_Method_Risk`: payment method is open account.
- `Smurfing_Risk`: vendor has many small transactions, which can suggest structuring.

A transaction is labelled `Suspicious` when its total suspicion score is at least 2. This keeps the rule simple and explainable: one warning sign alone may be normal, but multiple warning signs together deserve review.

## Distribution 1: Skellam distribution

The Skellam distribution is a discrete distribution for the difference between two count variables. If two independent variables follow Poisson distributions, then their difference follows a Skellam distribution.

Example:

```text
K = X_1 - X_2
```

where `X_1` and `X_2` are count variables. This is useful when the question is not simply "how many events happened?", but "what is the net difference between two event counts?"

Creative use case: in trade analysis, Skellam-style thinking can be useful for comparing two count processes such as reported shipments from one country versus received shipments in another country. A large unexpected difference may signal missing records, timing delays, or suspicious reporting.

It is also useful in sports scores, image noise, and any problem where the difference between two event counts matters more than either count alone. SciPy documents Skellam as a discrete random variable and notes its connection to Poisson random variables.

## Distribution 2: Tweedie distribution

The Tweedie distribution is less familiar because it is actually a family of distributions. It is often used when the data has many exact zeros but the non-zero values are positive and continuous.

This makes it interesting for real-world financial and risk data. For example, many customers may have zero insurance claims, but the customers who do make claims can have widely varying positive claim amounts. A normal distribution is not suitable because it allows negative values, and a simple Poisson distribution is not suitable because claim amounts are continuous.

The Tweedie family is used in generalized linear models. A key idea is the variance power parameter, which controls how the variance changes with the mean. Statsmodels describes Tweedie as a GLM family with a variance power parameter.

Creative use case: for shipment-risk analysis, a Tweedie model could describe loss or penalty amounts where many shipments have zero loss, but problematic shipments have positive loss values with a long right tail.

## Comparison

Skellam is useful for integer differences, such as count A minus count B. Tweedie is useful for non-negative amounts that combine many zeros with positive continuous values. Both are more specialized than the usual normal or binomial distributions, and both match practical risk-analysis problems better than a one-size-fits-all model.

## Sources

- SciPy documentation for Skellam distribution: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skellam.html
- Original Skellam paper DOI listed in SciPy docs: https://doi.org/10.2307/2981372
- Statsmodels documentation for Tweedie family: https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.family.Tweedie.html
- Scikit-learn TweedieRegressor documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html
