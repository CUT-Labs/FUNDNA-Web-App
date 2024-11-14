from scipy.integrate import odeint
from scipy.optimize import minimize, Bounds
import numpy as np


class ODESolver:
    def __init__(self, translation_scheme):
        self.translation_scheme = translation_scheme
        self.initial_conditions = list(translation_scheme.get_initial_conditions().values())
        self.rate_constants, self.ode_strings, self.ode_functions = translation_scheme.generate_odes()

    def crn_odes(self, y, t, rate_values):
        # Define time-dependent constants for X0 and X1
        X0 = 1 - t
        X1 = t

        # Map concentrations to dictionary format for lambda evaluation
        y_dict = dict(zip(self.translation_scheme.get_initial_conditions().keys(), y))
        y_dict.update({"X_0": X0, "X_1": X1})  # Add X0 and X1 to the dictionary

        # Update rate constant dictionary with current rate values
        k_dict = {f"k_r{i}": rate_values[i] for i in range(len(rate_values))}

        # Calculate the derivative for each ODE function using current rate constants
        dydt = [ode_func(y_dict, t, k_dict) for ode_func in self.ode_functions]

        return dydt

    def solve_odes(self, rate_values, time_points):
        # Solve the ODE system with the provided rate constants
        solution = odeint(self.crn_odes, self.initial_conditions, time_points, args=(rate_values,))
        return solution

    def objective_function(self, rate_values, time_points, target_function):
        # Solve ODEs with the current rate values
        solution = self.solve_odes(rate_values, time_points)
        Y0, Y1 = solution[:, -2], solution[:, -1]  # Assuming last two columns are Y0 and Y1

        # Calculate the Y_ratio as the output of interest
        with np.errstate(divide='ignore', invalid='ignore'):
            Y_ratio = np.where(Y0 + Y1 == 0, 0, Y0 / (Y0 + Y1))

        # Calculate target values based on the target function
        target_values = target_function(time_points)

        # Compute Mean Squared Error (MSE) between Y_ratio and target_values
        mse = np.mean((Y_ratio - target_values) ** 2)
        return mse

    def optimize_rate_constants(self, time_points, target_function):
        # Supported methods that handle bounds
        methods = ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']

        # Initial guesses for rate constants
        initial_guess = list(self.rate_constants.values())
        bounds = Bounds(0, 10**10)  # Apply bounds for each rate constant

        # Dictionary to store results for each method
        results = {}

        # Try each method and store the optimized constants and error
        for method in methods:
            try:
                result = minimize(
                    self.objective_function,
                    initial_guess,
                    args=(time_points, target_function),
                    method=method,
                    bounds=bounds
                )

                # Map optimized rate constants to their names
                optimized_rate_constants = {f"k_r{i}": result.x[i] for i in range(len(result.x))}
                error = result.fun
                results[method] = (optimized_rate_constants, error)

            except Exception as e:
                # Log failed attempts with an error placeholder
                print(f"Method {method} failed with error: {e}")
                results[method] = (None, float("inf"))

        return results
