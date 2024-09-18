import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib

#matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


# Define the ODE system
def crn_odes(y, t, k):
    S0, S1, Y0, Y1 = y
    X0 = 1 - t  # Fractional encoding: [X0] + [X1] = 1
    X1 = t

    # Unpack rate constants
    k1, k2, k3, k4, k5, k6, k7, k8 = k

    # ODEs based on the CRN
    dS0_dt = - (k3 * S0 * X1 + k4 * S0 * X1) + (k5 * S1 * X0 + k6 * S1 * X0)
    dS1_dt = - (k5 * S1 * X0 + k6 * S1 * X0) + (k3 * S0 * X1 + k4 * S0 * X1)
    dY0_dt = k1 * S0 * X0 + k3 * S0 * X1 + k5 * S1 * X0 + k7 * S1 * X1
    dY1_dt = k2 * S0 * X0 + k4 * S0 * X1 + k6 * S1 * X0 + k8 * S1 * X1

    return [dS0_dt, dS1_dt, dY0_dt, dY1_dt]


# Initial conditions
S0_0 = 0.01
S1_0 = 0.0
Y0_0 = 0.0
Y1_0 = 0.0
y0 = [S0_0, S1_0, Y0_0, Y1_0]

# Time points to solve over
t = np.linspace(0, 1, 1000)  # Since X0 + X1 = 1, vary t to represent different [X0], [X1]


# Original function f(x) = 0.5x
def original_function(x):
    return 0.5*x


# Objective function to minimize (Mean Squared Error)
def objective_function(k):
    # Solve the ODEs with the current k values
    solution = odeint(crn_odes, y0, t, args=(k,))
    S0, S1, Y0, Y1 = solution.T

    # Calculate the actual Y_ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        Y_ratio = np.where(Y0 + Y1 == 0, 0, Y0 / (Y0 + Y1))  # Handle divide by zero

    X_ratio = 1 - t  # X0 / (X0 + X1)

    # Calculate the original function values at the same X_ratio points
    original_values = original_function(X_ratio)

    # Compute the MSE between the actual Y_ratio and the original function
    mse = np.mean((Y_ratio - original_values) ** 2)

    return mse

s0_out = 0.5
s1_out = 0.5

# Initial guess for the rate constants
initial_k = [1-s0_out,  #s0 -> s0 *
             s0_out,    #s0 -> s0 (true output)
             1-s1_out,  #s0 -> s1 *
             s1_out,    #s0 -> s1 (true output)

             1-s0_out,  #s1 -> s0 *
             s0_out,    #s1 -> s0 (true output)
             1-s1_out,  #s1 -> s1 *
             s1_out]    #s1 -> s1 (true output)

# Optimize the rate constants to minimize the MSE
result = minimize(objective_function, initial_k, method='Nelder-Mead')

# Extract the optimized rate constants
optimized_k = result.x

# Print the optimized rate constants
print("Optimized rate constants:")
print(optimized_k)

# Solve the ODEs with the optimized rate constants
solution = odeint(crn_odes, y0, t, args=(optimized_k,))
S0, S1, Y0, Y1 = solution.T

# Calculate the Y_ratio and X_ratio for plotting
with np.errstate(divide='ignore', invalid='ignore'):
    Y_ratio = np.where(Y0 + Y1 == 0, 0, Y0 / (Y0 + Y1))  # Handle divide by zero

X_ratio = 1 - t  # X0 / (X0 + X1)

# Calculate the original function for plotting
original_values = original_function(X_ratio)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(X_ratio, Y_ratio, label='Optimized Y0/(Y0+Y1)')
plt.plot(X_ratio, original_values, label='Original f(x) = 0.5*x', linestyle='dashed')
plt.xlabel('X0 / (X0 + X1)')
plt.ylabel('Y0 / (Y0 + Y1)')
plt.title('Comparison of Optimized Y0/(Y0+Y1) and Original Function')
plt.legend()
plt.grid(True)
plt.show()

# Print the final MSE
final_mse = objective_function(optimized_k)
print(f"Final Mean Squared Error: {final_mse}")
