# week4_least_squares.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class LinearRegressionScratch:
    """
    Linear Regression implemented from scratch using ONLY NumPy
    No scikit-learn - pure matrix operations
    """
    
    def __init__(self, method='normal', learning_rate=0.01, iterations=1000):
        """
        method: 'normal' (closed form) or 'gradient' (iterative)
        """
        self.method = method
        self.alpha = learning_rate
        self.iterations = iterations
        self.theta = None
        self.cost_history = []
        self.feature_means = None
        self.feature_stds = None
    
    def add_bias(self, X):
        """
        Add bias column (column of 1s) to feature matrix
        X: m x n matrix -> returns m x (n+1) matrix
        """
        m = X.shape[0]
        return np.c_[np.ones(m), X]  # Prepend column of 1s
    
    def normalize_features(self, X):
        """
        Feature normalization (z-score) for gradient descent
        """
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0)
        # Avoid division by zero
        self.feature_stds[self.feature_stds == 0] = 1
        return (X - self.feature_means) / self.feature_stds
    
    def fit(self, X, y):
        """
        Fit linear regression
        X: feature matrix (m x n)
        y: target vector (m x 1)
        """
        print(f"\n{'='*60}")
        print(f"FITTING LINEAR REGRESSION")
        print(f"Method: {self.method}")
        print(f"{'='*60}")
        
        # Store original dimensions
        m, n = X.shape
        print(f"Training samples (m): {m}")
        print(f"Features (n): {n}")
        
        # Add bias column
        X_b = self.add_bias(X)
        print(f"Design matrix shape: {X_b.shape}")
        
        if self.method == 'normal':
            # NORMAL EQUATION: θ = (X^T X)^(-1) X^T y
            print("\n--- Normal Equation Method ---")
            
            # Step 1: Compute X^T X
            XtX = X_b.T @ X_b
            print(f"X^T X shape: {XtX.shape}")
            
            # Step 2: Check if invertible (determinant)
            det = np.linalg.det(XtX)
            print(f"Determinant of X^T X: {det:.6f}")
            
            if abs(det) < 1e-10:
                print("WARNING: X^T X is near-singular! Using pseudo-inverse.")
                XtX_inv = np.linalg.pinv(XtX)
            else:
                # Step 3: Compute inverse
                XtX_inv = np.linalg.inv(XtX)
            
            # Step 4: Compute X^T y
            Xty = X_b.T @ y
            print(f"X^T y shape: {Xty.shape}")
            
            # Step 5: Final computation
            self.theta = XtX_inv @ Xty
            print(f"\n✅ Parameters computed: θ = (X^T X)^(-1) X^T y")
            
        elif self.method == 'gradient':
            # GRADIENT DESCENT
            print("\n--- Gradient Descent Method ---")
            
            # Normalize features for better convergence
            X_norm = self.normalize_features(X)
            X_b = self.add_bias(X_norm)
            
            # Initialize parameters
            self.theta = np.zeros(n + 1)
            
            print(f"Initial θ: {self.theta}")
            print(f"Learning rate (α): {self.alpha}")
            print(f"Iterations: {self.iterations}")
            
            for i in range(self.iterations):
                # Hypothesis: h = Xθ
                h = X_b @ self.theta
                
                # Error: e = h - y
                error = h - y
                
                # Gradient: (1/m) * X^T * error
                gradient = (1/m) * (X_b.T @ error)
                
                # Update: θ := θ - α * gradient
                self.theta = self.theta - self.alpha * gradient
                
                # Cost for monitoring
                cost = (1/(2*m)) * np.sum(error ** 2)
                self.cost_history.append(cost)
                
                if i % 100 == 0:
                    print(f"  Iteration {i}: Cost = {cost:.6f}")
            
            print(f"\n✅ Gradient descent converged")
            print(f"Final cost: {self.cost_history[-1]:.6f}")
        
        print(f"\nLearned parameters (θ):")
        for i, param in enumerate(self.theta):
            if i == 0:
                print(f"  θ₀ (bias): {param:.6f}")
            else:
                print(f"  θ{i}: {param:.6f}")
        
        return self
    
    def predict(self, X):
        """
        Make predictions using learned parameters
        """
        if self.theta is None:
            raise ValueError("Model not fitted yet!")
        
        # Normalize if using gradient descent
        if self.method == 'gradient' and self.feature_means is not None:
            X = (X - self.feature_means) / self.feature_stds
        
        X_b = self.add_bias(X)
        return X_b @ self.theta
    
    def score(self, X, y):
        """
        R-squared score: 1 - (SS_res / SS_tot)
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_equation(self, feature_names=None):
        """
        Return the regression equation as a string
        """
        if self.theta is None:
            return "Model not fitted"
        
        terms = [f"{self.theta[0]:.4f}"]
        for i in range(1, len(self.theta)):
            name = feature_names[i-1] if feature_names else f"x{i}"
            terms.append(f"{self.theta[i]:+.4f}·{name}")
        
        return "y = " + " ".join(terms)


# ============================================
# DEMONSTRATION WITH METALS TRADING DATA
# ============================================

def create_sample_metals_data():
    """
    Create synthetic metals trading data for regression
    Features: Volume, Market Price, Country Risk Score
    Target: Unit Price (what we predict)
    """
    np.random.seed(42)
    m = 1000  # 1000 transactions
    
    # Feature 1: Volume (MT) - larger volumes get discounts
    volume = np.random.lognormal(5, 1, m)  # ~100-500 MT typical
    
    # Feature 2: Market Spot Price ($/MT)
    market_price = np.random.normal(2000, 500, m)
    
    # Feature 3: Country Risk Score (0-10, higher = riskier)
    country_risk = np.random.uniform(0, 10, m)
    
    # True relationship (with noise):
    # Unit Price = 0.95 * Market Price - 0.1 * Volume + 5 * Risk + noise
    true_theta = np.array([0, 0.95, -0.1, 5.0])
    
    X = np.column_stack([volume, market_price, country_risk])
    X_b = np.c_[np.ones(m), X]
    
    noise = np.random.normal(0, 50, m)
    y = X_b @ true_theta + noise
    
    return X, y, ['Volume_MT', 'Market_Price', 'Country_Risk']


def demonstrate_normal_equation():
    """
    Step-by-step demonstration of Normal Equation
    """
    print("=" * 70)
    print("WEEK 4: LEAST SQUARES REGRESSION - NORMAL EQUATION")
    print("=" * 70)
    
    # Create data
    X, y, feature_names = create_sample_metals_data()
    
    print(f"\n📊 Dataset:")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]} ({', '.join(feature_names)})")
    print(f"  Target: Unit_Price_USD")
    
    # Fit using Normal Equation
    model = LinearRegressionScratch(method='normal')
    model.fit(X, y)
    
    # Results
    print(f"\n📈 Results:")
    print(f"  R² Score: {model.score(X, y):.4f}")
    print(f"  Equation: {model.get_equation(feature_names)}")
    
    # Predictions vs Actual
    y_pred = model.predict(X)
    print(f"\n🔮 Sample Predictions:")
    for i in range(5):
        print(f"  Actual: ${y[i]:.2f}, Predicted: ${y_pred[i]:.2f}, Error: ${abs(y[i]-y_pred[i]):.2f}")
    
    return model


def demonstrate_gradient_descent():
    """
    Step-by-step demonstration of Gradient Descent
    """
    print("\n" + "=" * 70)
    print("WEEK 4: LEAST SQUARES REGRESSION - GRADIENT DESCENT")
    print("=" * 70)
    
    # Create data
    X, y, feature_names = create_sample_metals_data()
    
    # Fit using Gradient Descent
    model = LinearRegressionScratch(method='gradient', learning_rate=0.1, iterations=1000)
    model.fit(X, y)
    
    # Plot cost history
    plt.figure(figsize=(10, 6))
    plt.plot(model.cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost J(θ)')
    plt.title('Gradient Descent: Cost Function Convergence')
    plt.grid(True)
    plt.savefig('gradient_descent_convergence.png')
    print(f"\n📊 Plot saved: gradient_descent_convergence.png")
    
    # Results
    print(f"\n📈 Results:")
    print(f"  R² Score: {model.score(X, y):.4f}")
    print(f"  Equation: {model.get_equation(feature_names)}")
    
    return model


def compare_methods():
    """
    Compare Normal Equation vs Gradient Descent
    """
    print("\n" + "=" * 70)
    print("COMPARISON: NORMAL EQUATION vs GRADIENT DESCENT")
    print("=" * 70)
    
    X, y, feature_names = create_sample_metals_data()
    
    # Normal Equation
    model_normal = LinearRegressionScratch(method='normal')
    model_normal.fit(X, y)
    r2_normal = model_normal.score(X, y)
    
    # Gradient Descent
    model_grad = LinearRegressionScratch(method='gradient', learning_rate=0.1, iterations=1000)
    model_grad.fit(X, y)
    r2_grad = model_grad.score(X, y)
    
    print(f"\n{'='*70}")
    print(f"COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'R² Score':<15} {'Time Complexity':<20}")
    print(f"{'-'*70}")
    print(f"{'Normal Equation':<25} {r2_normal:<15.4f} {'O(n³) - slow for large n':<20}")
    print(f"{'Gradient Descent':<25} {r2_grad:<15.4f} {'O(kn²) - scalable':<20}")
    
    print(f"\n📌 When to use which:")
    print(f"  • Normal Equation: n < 10,000 features (exact solution)")
    print(f"  • Gradient Descent: n > 10,000 features, online learning, regularization")
    
    # Parameter comparison
    print(f"\n📊 Parameter Comparison:")
    print(f"{'Parameter':<15} {'Normal':<15} {'Gradient':<15} {'Difference':<15}")
    for i in range(len(model_normal.theta)):
        name = f"θ{i}" if i == 0 else feature_names[i-1]
        diff = abs(model_normal.theta[i] - model_grad.theta[i])
        print(f"{name:<15} {model_normal.theta[i]:<15.6f} {model_grad.theta[i]:<15.6f} {diff:<15.6f}")


# ============================================
# MATHEMATICAL DERIVATION DOCUMENTATION
# ============================================

def print_derivation():
    """
    Print the complete mathematical derivation
    """
    derivation = """
================================================================================
WEEK 4: MATHEMATICAL DERIVATION - LEAST SQUARES IN VECTOR FORM
================================================================================

PROBLEM SETUP:
--------------
Given m training examples, each with n features:
  x⁽ⁱ⁾ = [x₁⁽ⁱ⁾, x₂⁽ⁱ⁾, ..., xₙ⁽ⁱ⁾]  (feature vector for i-th example)
  y⁽ⁱ⁾ = target value

We want to find parameters θ = [θ₀, θ₁, ..., θₙ] that minimize:

  J(θ) = (1/2m) Σᵢ₌₁ᵐ (hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²

where hθ(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ

MATRIX NOTATION:
----------------
Design matrix X (m × (n+1)):
  X = | 1  x₁⁽¹⁾  x₂⁽¹⁾  ...  xₙ⁽¹⁾ |
      | 1  x₁⁽²⁾  x₂⁽²⁾  ...  xₙ⁽²⁾ |
      | ...                          |
      | 1  x₁⁽ᵐ⁾  x₂⁽ᵐ⁾  ...  xₙ⁽ᵐ⁾ |

Parameter vector θ ((n+1) × 1):
  θ = [θ₀, θ₁, θ₂, ..., θₙ]ᵀ

Target vector y (m × 1):
  y = [y⁽¹⁾, y⁽²⁾, ..., y⁽ᵐ⁾]ᵀ

Hypothesis vector:
  h = Xθ  (m × 1)

COST FUNCTION IN VECTOR FORM:
-----------------------------
  J(θ) = (1/2m) (Xθ - y)ᵀ (Xθ - y)

EXPANDING:
  J(θ) = (1/2m) [θᵀXᵀXθ - 2θᵀXᵀy + yᵀy]

TAKING DERIVATIVE:
  ∂J/∂θ = (1/m) [XᵀXθ - Xᵀy] = 0

SETTING TO ZERO (NORMAL EQUATION):
  XᵀXθ = Xᵀy

  θ = (XᵀX)⁻¹ Xᵀy    ← CLOSED FORM SOLUTION

GRADIENT DESCENT (ITERATIVE):
  θ := θ - α · (1/m) Xᵀ(Xθ - y)

where α = learning rate

================================================================================
WHY THIS MATTERS FOR YOUR PROJECT:
================================================================================

1. PREDICTING UNIT PRICES:
   Given Volume, Market Price, Country Risk → predict fair Unit Price
   Flag transactions where actual price deviates >20% from predicted

2. ANOMALY DETECTION:
   Large residuals (|y - ŷ|) indicate potential fraud/under-invoicing

3. RISK SCORING:
   Regression coefficients (θ) show which features most affect price
   - Large θ for Country Risk = high-risk vendors charge more
   - Negative θ for Volume = bulk discounts exist

4. MULTIVARIATE DETECTION:
   n-dimensional input captures multiple fraud indicators simultaneously
================================================================================
"""
    print(derivation)
    with open('week4_derivation.txt', 'w') as f:
        f.write(derivation)
    print("💾 Saved to week4_derivation.txt")


# ============================================
# RUN EVERYTHING
# ============================================
if __name__ == "__main__":
    # 1. Print mathematical derivation
    print_derivation()
    
    # 2. Demonstrate Normal Equation
    model_normal = demonstrate_normal_equation()
    
    # 3. Demonstrate Gradient Descent
    model_grad = demonstrate_gradient_descent()
    
    # 4. Compare methods
    compare_methods()
    
    print("\n" + "=" * 70)
    print("✅ WEEK 4 COMPLETE!")
    print("=" * 70)
    print("\n📁 Files generated:")
    print("  • week4_derivation.txt - Full mathematical derivation")
    print("  • gradient_descent_convergence.png - Cost function plot")
    print("  • week4_report.txt - Summary report (generate below)")