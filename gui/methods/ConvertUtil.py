import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for rendering graphs in Django
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
import sympy as sp
from sympy.parsing.latex import parse_latex


def generatePoints(function, x_bounds=[0, 1], numPoints=1000):
    # Generate x values between x_bounds[0] and x_bounds[1]
    x = np.linspace(x_bounds[0], x_bounds[1], numPoints)

    # Apply the lambda function to generate y values
    y = function(x)

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
    fig.savefig(buf, format='svg')
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

    # Define the variable(s)
    x = sp.symbols('x')  # assuming the LaTeX expression uses 'x' as the variable

    # Create a lambda function from the SymPy expression
    lambda_function = sp.lambdify(x, sympy_expr, modules=['numpy'])

    return lambda_function
