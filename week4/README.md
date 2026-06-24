# Week 4 Report

## Task status

The Week 3 labelling task has been completed. The transaction file `metallurgical_ledgers.xlsx` was labelled using simple rule-based criteria such as sanctioned-country risk, price deviation, high transaction value, round-number values, open-account payment, and possible smurfing behaviour. The labelled output was saved as `week3/labelled_metallurgical_ledgers.xlsx`.

## Least squares method

Least squares is a method for fitting a line or linear model when the data points do not lie exactly on one line. Instead of trying to pass through every point, it chooses the line that makes the total squared error as small as possible.

For one input variable, the fitted line has the form:

```text
y_hat = beta_0 + beta_1 x
```

For n-dimensional input, the fitted model becomes:

```text
y_hat = beta_0 + beta_1 x_1 + beta_2 x_2 + ... + beta_n x_n
```

In vector form, this is written as:

```text
y_hat = X beta
```

where `X` is the design matrix, `beta` is the vector of coefficients, and `y` is the observed output vector. The least squares estimate minimizes:

```text
J(beta) = ||X beta - y||_2^2
```

Taking the derivative and setting it equal to zero gives the normal equations:

```text
X^T X beta = X^T y
```

If `X^T X` is invertible, the solution is:

```text
beta_hat = (X^T X)^(-1) X^T y
```

This means the best-fitting line is found by projecting the observed vector `y` onto the column space of `X`. The residual error is orthogonal to the fitted space, which is why the method has a clean geometric interpretation.

## Why it matters

Least squares is important because it is the foundation of linear regression. It can be used to estimate trends, predict values, and measure relationships between variables. In trade or transaction analysis, it could help model expected shipment delay, expected transaction value, or normal price behaviour, and then flag observations that are far away from the fitted trend.

## Sources

- Stanford EE263 lecture notes on least squares: https://see.stanford.edu/materials/lsoeldsee263/05-ls.pdf
- Penn State STAT 501 Regression Methods notes: https://online.stat.psu.edu/stat501/
