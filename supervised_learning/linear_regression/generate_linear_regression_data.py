import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 500 random x values between 0.5 and 4.0 (1000 sqft to 4000 sqft)
x_train = np.random.uniform(0.5, 2.5, 500)

# Assume true relationship: price = 150 * size + 120 + noise
noise = np.random.normal(0, 25, 500)
y_train = 150 * x_train + 120 + noise

# Save to .npz file
np.savez('linear_regression_data.npz', x_train=x_train, y_train=y_train)
print("Saved 500 examples to linear_regression_data.npz")
