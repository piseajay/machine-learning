import numpy as np
import matplotlib.pyplot as plt

try:
    plt.style.use(
        "/Users/ajaypise/workspace/Machine_Learning/machine_learning/deeplearning.mplstyle"
    )
except OSError:
    print("Warning: deeplearning.mplstyle not found. Using default style.")

# Load 500-example dataset
data = np.load('./supervised_learning/linear_regression_data.npz')
x_train = data['x_train']
y_train = data['y_train']

print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")

i = 0  # Change this to 1 to see (x^1, y^1)

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()

w = 166
b = 90
print(f"w: {w}")
print(f"b: {b}")


def comput_model_output(x: np.ndarray, w: float, b: float):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      y (ndarray (m,)): target values
    """
    m = x.shape[0]
    f_wb = np.zeros(m)

    for i in range(m):
        f_wb[i] = (w * x[i]) + b
    
    return f_wb

tmp_f_wb = comput_model_output(x=x_train, w=w, b=b)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()