"""
Linear Regression Example using California Housing Dataset
This dataset is based on California census data and is commonly used for regression tasks.
Dataset source: Sklearn datasets (originally from StatLib repository)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Load the California Housing dataset
print("Loading California Housing dataset...")
housing = fetch_california_housing()

# Create a DataFrame for better visualization
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Target'] = housing.target

print("\n" + "="*60)
print("DATASET INFORMATION")
print("="*60)
print(f"\nDataset shape: {df.shape}")
print(f"Number of samples: {len(df)}")
print(f"Number of features: {len(housing.feature_names)}")
print(f"\nFeatures: {housing.feature_names}")
print(f"\nTarget: Median house value (in $100,000s)")

print("\n" + "="*60)
print("FIRST 5 ROWS")
print("="*60)
print(df.head())

print("\n" + "="*60)
print("STATISTICAL SUMMARY")
print("="*60)
print(df.describe())

print("\n" + "="*60)
print("MISSING VALUES")
print("="*60)
print(df.isnull().sum())

# Prepare features and target
X = housing.data
y = housing.target

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n" + "="*60)
print("DATA SPLIT")
print("="*60)
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# Feature scaling for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Linear Regression model
print("\n" + "="*60)
print("TRAINING LINEAR REGRESSION MODEL")
print("="*60)
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("Model training completed!")

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Evaluate the model
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

# Training metrics
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print("\nTraining Set Metrics:")
print(f"  Mean Squared Error (MSE):  {train_mse:.4f}")
print(f"  Root Mean Squared Error (RMSE): {train_rmse:.4f}")
print(f"  Mean Absolute Error (MAE): {train_mae:.4f}")
print(f"  R² Score: {train_r2:.4f}")

# Testing metrics
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nTesting Set Metrics:")
print(f"  Mean Squared Error (MSE):  {test_mse:.4f}")
print(f"  Root Mean Squared Error (RMSE): {test_rmse:.4f}")
print(f"  Mean Absolute Error (MAE): {test_mae:.4f}")
print(f"  R² Score: {test_r2:.4f}")

# Visualizations
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# 1. Actual vs Predicted values
plt.scatter(y_test, y_test_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Values ($100k)', fontsize=12)
plt.ylabel('Predicted Values ($100k)', fontsize=12)
plt.title('Actual vs Predicted Values (Test Set)', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/ajaypise/workspace/Machine_Learning/machine_learning/supervised_learning/linear_regression/linear_regression_analysis.png', 
            dpi=300, bbox_inches='tight')
print("Plots saved to 'linear_regression_analysis.png'")
plt.show()
