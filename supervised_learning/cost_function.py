import numpy as np
import matplotlib.pyplot as plt

def compute_cost(x, y, w, b):
    """
    Compute cost for linear regression using the formula:
    J(w, b) = (1/2m) * sum((f_wb(x(i)) - y(i))^2)
    where f_wb(x) = w*x + b
    """
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2
    total_cost = cost / (2 * m)
    return total_cost

# Example data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Generate a grid of w and b values for plotting
w_values = np.linspace(0, 3, 50)
b_values = np.linspace(-2, 2, 50)
W, B = np.meshgrid(w_values, b_values)
J = np.zeros_like(W)

# Compute cost for each (w, b) pair
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        J[i, j] = compute_cost(X, y, W[i, j], B[i, j])

# Plotting the cost function
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, B, J, cmap='viridis')
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Cost J(w, b)')
ax.set_title('Cost Function Surface')
plt.show()