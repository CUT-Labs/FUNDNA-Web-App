# main.py
from TranslationScheme import get_translation_scheme
from ODESolver import ODESolver
import numpy as np
from tabulate import tabulate

from io import BytesIO
import os
import webbrowser
import random
import string


import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # Use the Agg backend for rendering graphs in Django


# Graph generator with error metrics
def dualGraphWithErrors(x_values, y_values, expected_values, title, expectedName, func2Name, pltVariable=False):
    averageError = np.mean(y_values - expected_values)
    std_dev = np.std(y_values - expected_values)
    mse = np.mean((y_values - expected_values) ** 2)
    mae = np.mean(np.abs(y_values - expected_values))

    error_labels = ['Average Error', 'Error Std. Dev.', 'MSE', 'MAE']
    error_values = [averageError, std_dev, mse, mae]

    fig, (ax_main, ax_caption) = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 8))

    # Plot results
    ax_main.plot(x_values, expected_values, label=expectedName, color='blue')
    ax_main.plot(x_values, y_values, label=func2Name, linestyle='--', color='orange')
    ax_main.set_xlabel('x', fontsize=16)
    ax_main.set_ylabel('f(x)', fontsize=16)
    ax_main.set_title(title, fontsize=20)
    ax_main.legend(fontsize=16)
    ax_main.tick_params(axis='both', labelsize=14)
    ax_main.grid(True)

    # Caption with error metrics
    caption = ";   ".join([f"{label}: {value:.3E}" for label, value in zip(error_labels, error_values)])
    ax_caption.text(0.5, 0.5, caption, ha='center', va='center', fontsize=14)
    ax_caption.axis('off')

    plt.tight_layout()

    if pltVariable:
        return plt

    buf = BytesIO()
    fig.savefig(buf, format='svg')
    buf.seek(0)
    plt.close(fig)
    return buf


def ShowGraphBuffer(buf, show=False, temp=False):
    # Generate a random file name
    random_filename = ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + '.svg'
    temp_filename = os.path.join(os.path.dirname(__file__), random_filename)

    # Write buffer contents to the file
    with open(temp_filename, 'wb') as temp_file:
        temp_file.write(buf.read())

    if show:
        # Open the SVG file in the default web browser
        webbrowser.open(f'file://{temp_filename}')

    if temp:
        # Remove the file after opening
        os.remove(temp_filename)


def main():
    # Define the target function for comparison
    def original_function(x):
        return 0.5 * x + 0.25

    # Define states with example probabilities for each state output
    states = [0.25, 0.75]

    # Initialize the translation scheme
    scheme_name = "JPHuseProposed"
    scheme = get_translation_scheme(scheme_name, states)

    # Set up time points for solving the ODEs
    time_points = np.linspace(0, 1, 1000)  # Time range for X0 and X1 sweep

    # Initialize the ODE solver
    solver = ODESolver(scheme)

    # Optimize rate constants to minimize error
    optimized_results = solver.optimize_rate_constants(time_points, original_function)

    # Tabulate MSE scores for each method
    results_table = []
    min_error = float('inf')
    best_method = None
    best_constants = None

    for method, (optimized_rate_constants, error) in optimized_results.items():
        if optimized_rate_constants is not None:
            results_table.append([method, error])
            if error < min_error:
                min_error = error
                best_method = method
                best_constants = optimized_rate_constants

    print("\n=== Optimization Results: MSE Scores for Each Method ===")
    print(tabulate(results_table, headers=["Method", "Error (MSE)"], floatfmt=".6f"))

    # Plot for the best-performing method
    if best_constants is not None:
        solution = solver.solve_odes(list(best_constants.values()), time_points)
        Y0, Y1 = solution[:, -2], solution[:, -1]

        # Handle NaN in Y_ratio by replacing with zero
        with np.errstate(divide='ignore', invalid='ignore'):
            Y_ratio = np.where(np.isnan(Y0 / (Y0 + Y1)), 0, Y0 / (Y0 + Y1))

        expected_values = original_function(time_points)

        title = f"Best Method: {best_method} - Error (MSE): {min_error:.6f}"
        buf = dualGraphWithErrors(time_points, Y_ratio, expected_values, title, "Expected", "Best Fit")

        ShowGraphBuffer(buf)

        print("\n=== Best Performing Method ===")
        print(f"Method: {best_method}")
        for rate_name, value in best_constants.items():
            print(f"  {rate_name} = {value}")
        print(f"  Error (MSE) = {min_error}\n")


if __name__ == "__main__":
    main()
