import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from math import *

from io import BytesIO
import base64

import sympy as sp
from sympy.parsing.latex import parse_latex

from gui.classes import *
from gui.methods import *

matplotlib.use('Agg')  # Use the Agg backend for rendering graphs in Django


def generatePoints(function, x_bounds=[0, 1], numPoints=1000):
    """
    Generates x and y points for a given function.

    Args:
    - function: Lambda function
    - x_bounds: Range for the x-axis
    - numPoints: Number of points to generate

    Returns:
    - x: numpy array of x points
    - y: numpy array of corresponding y points
    """
    print("Generating x value linspace")
    # Generate x values between x_bounds[0] and x_bounds[1]
    x = np.linspace(x_bounds[0], x_bounds[1], numPoints)

    # Apply the lambda function to generate y values
    try:
        print("Applying lambda function to x value linspace")
        print(f'x: {type(x)}')
        print(f'function: {type(function)}')
        y = function(x)
        print(f'y: {type(y)}')
    except Exception as e:
        raise ValueError(f"Error evaluating function: {str(e)}")

    return x, y


def generateGraph(x, y):
    """
    Generates a graph for a given function.

    Args:
    - x: numpy array of x points
    - y: numpy array of associated y points

    Returns:
    - buf: BytesIO object containing the plot image.
    """
    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(x, y, label=f"f(x)", color='blue')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Plot of the Function')
    ax.grid(True)
    ax.legend()

    # Save plot to a BytesIO object
    buf = BytesIO()
    fig.savefig(buf, format='svg')  # Save as SVG for better scalability
    buf.seek(0)

    plt.close(fig)  # Close the figure to free up memory

    return buf


def LatexToLambda(latex_expr):
    """
    Convert a LaTeX expression into a Python lambda function using SymPy.

    Args:
    - latex_expr: A string representing a LaTeX function (e.g., r'\\frac{1}{x} + \\sin{x}').

    Returns:
    - A Python lambda function that evaluates the LaTeX expression.
    """
    # Parse the LaTeX string to a SymPy expression
    sympy_expr = parse_latex(latex_expr)

    # Replace mathematical constants (like 'e') with their numeric equivalents
    sympy_expr = sympy_expr.evalf()

    # Define the variable(s)
    x = sp.symbols('x')  # assuming the LaTeX expression uses 'x' as the variable

    # Create a lambda function from the SymPy expression
    try:
        lambda_function = sp.lambdify(x, sympy_expr, modules=['numpy'])
    except Exception as e:
        raise ValueError(f"Error converting LaTeX to lambda: {str(e)}")

    return lambda_function


def functionData(latex_input):
    # Convert LaTeX to lambda and generate points
    function_lambda = LatexToLambda(latex_input)
    x_values, y_values = generatePoints(function_lambda)

    # Generate graph image
    graph = generateGraph(x_values, y_values)

    # Encode graph in base64 for rendering in the template
    graph_url = "data:image/svg+xml;base64," + base64.b64encode(graph.getvalue()).decode()

    return function_lambda, graph_url


def determineFunctionType(functionStr):
    # Convert to Function Parameter Types
    print("functionstr:\t", functionStr)
    funcType = None
    if functionStr.__contains__("log"):
        funcType = FuncTypes.LOGARITHMIC
    elif functionStr.__contains__("exp"):
        funcType = FuncTypes.EXPONENTIAL
    elif functionStr.__contains__("sin") or \
            functionStr.__contains__("cos") or \
            functionStr.__contains__("tan") or \
            functionStr.__contains__("csc") or \
            functionStr.__contains__("cot") or \
            functionStr.__contains__("sec") or \
            functionStr.__contains__("cosh") or \
            functionStr.__contains__("sinh") or \
            functionStr.__contains__("csch") or \
            functionStr.__contains__("coth") or \
            functionStr.__contains__("sech") or \
            functionStr.__contains__("tanh"):
        funcType = FuncTypes.SINUSOIDAL
    else:
        funcType = FuncTypes.POLYNOMIAL

    print("functype:\t", funcType.value)
    return funcType


def graphOriginalAndRearrangement(originalFunction, rearrangementLambda, pointEstimation, degreeEstimation):
    print("Graphing Original and Rearrangement")
    print("...generating original (expected) values")
    x, expected = generatePoints(originalFunction)
    print("...generating rearrangement (theoretical) values")
    _, rearrangement = generatePoints(rearrangementLambda)

    graph_io = dualGraphWithErrors(x,
                               rearrangement,
                               expected,
                               f'FUNDNA Approximation - Point: {pointEstimation} - Degree: {degreeEstimation}',
                               "Originanl Function",
                               "Rearrangement Function")

    return "data:image/svg+xml;base64," + base64.b64encode(graph_io.getvalue()).decode()


def dualGraphWithErrors(x_values, y_values, expected_values, title, expectedName, func2Name):
    averageError, std_dev, ratioUnder, mse, mae = analyzeError(y_values - expected_values)

    error_labels = ['x̄(E)', 'σ(E)', '%neg', 'MSE', 'MAE']
    error_values = [averageError, std_dev, ratioUnder * 100, mse, mae]

    # Create a figure with a main plot and a subplot for the caption
    fig, (ax_main, ax_caption) = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 8))

    # Plot the results in the main subplot
    ax_main.plot(x_values, expected_values, label=f'{expectedName}', color='blue')
    ax_main.plot(x_values, y_values, label=f'{func2Name}', linestyle='--', color='orange')
    ax_main.set_xlabel('x', fontsize=16)
    ax_main.set_ylabel('f(x)', fontsize=16)
    ax_main.set_title(f'{title}', fontsize=20)
    ax_main.legend(fontsize=16)
    ax_main.tick_params(axis='both', which='major', labelsize=14)  # Increase font size of axis ticks
    ax_main.grid(True)

    # Adding the figure caption in the caption subplot
    caption = ";   ".join([f"{label}: {value:.6f}" for label, value in zip(error_labels, error_values)])
    ax_caption.text(0.5, 0.5, caption, ha='center', va='center', fontsize=14)
    ax_caption.axis('off')  # Hide axis for the caption subplot

    plt.tight_layout()  # Adjust layout to prevent overlap

    # Save plot to a BytesIO object
    buf = BytesIO()
    fig.savefig(buf, format='svg')  # Save as SVG for better scalability
    buf.seek(0)

    plt.close(fig)  # Close the figure to free up memory

    return buf


def analyzeError(error):
    # Average Error
    averageError = np.mean(error)

    # Calculate standard deviation
    std_dev = np.std(error)

    # Percent negative errors
    ratioUnder = np.count_nonzero(np.array(error) < 0) / len(error)

    # Calculate mean squared error (MSE)
    mse = np.mean(np.square(error))

    # Calculate mean absolute error (MAE)
    mae = np.mean(np.abs(error))

    print(f"Average Error: \t{averageError}")
    print(f"Standard deviation of errors: \t{std_dev}")
    print(f"Ratio of negative errors: \t{ratioUnder}")
    print(f"Mean Squared Error (MSE): \t{mse}")
    print(f"Mean Absolute Error (MAE): \t{mae}")

    return averageError, std_dev, ratioUnder, mse, mae