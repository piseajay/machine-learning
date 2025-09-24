import numpy as np

# Example function: f(x) = (x - 3)^2
def f(x):
    return (x - 3) ** 2

# Derivative of the function: f'(x) = 2 * (x - 3)
def grad_f(x):
    return 2 * (x - 3)

def gradient_descent(starting_point, learning_rate, n_iterations):
    x = starting_point
    for i in range(n_iterations):
        grad = grad_f(x)
        x = x - learning_rate * grad
        print(f"Iteration {i+1}: x = {x}, f(x) = {f(x)}")
    return x

if __name__ == "__main__":
    initial_x = 0.0
    learning_rate = 0.1
    iterations = 10
    print("Running gradient descent...")
    final_x = gradient_descent(initial_x, learning_rate, iterations)
    print(f"\nFinal result: x = {final_x}, f(x) = {f(final_x)}")