import numpy as np
from cost_function import compute_cost

def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression.
    Args:
      x (ndarray (m,)): Data, m examples
      y (ndarray (m,)): target values
      w, b (scalar): model parameters
    Returns:
      dj_dw (scalar): gradient with respect to w
      dj_db (scalar): gradient with respect to b
    """
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    """
    Performs batch gradient descent to learn w and b.
    Args:
      x (ndarray (m,)): Data, m examples
      y (ndarray (m,)): target values
      w_in, b_in (scalar): initial values of w and b
      alpha (float): learning rate
      num_iters (int): number of iterations
    Returns:
      w (scalar): trained weight
      b (scalar): trained bias
      J_history (list): cost value at each iteration
    """
    w = w_in
    b = b_in
    J_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100:  # prevent resource exhaustion
            cost = compute_cost(x, y, w, b)
            J_history.append(cost)

        if i % (num_iters // 10) == 0 or i == num_iters - 1:
            print(f"Iteration {i:4d}: Cost {cost:.4f}, w: {w:.4f}, b: {b:.4f}")

    return w, b, J_history

# Example usage
if __name__ == "__main__":
    # Load data
    data = np.load('./supervised_learning/linear_regression_data.npz')
    x_train = data['x_train']
    y_train = data['y_train']

    # Initialize parameters
    w_init = 0
    b_init = 0
    iterations = 1000
    alpha = 0.01

    # Run gradient descent
    w_final, b_final, J_hist = gradient_descent(x_train, y_train, w_init, b_init, alpha, iterations)

    print(f"Trained parameters: w = {w_final}, b = {b_final}")