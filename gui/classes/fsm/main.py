import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob
from math import *
import random
from tabulate import tabulate

# Importing the FSM classes and the parse_file function
from FSMUtil import FSM, State, Transition, parse_file
from FunctionTypes import Types
from SolverUtil import FSMSolver
from Bernstein import Bernstein

#matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


def generatePlot(x_values, y_values, expected_values, fsm):
    averageError, std_dev, ratioUnder, mse, mae = analyzeError(fsm.error)

    error_labels = ['x̄(E)', 'σ(E)', '%neg', 'MSE', 'MAE']
    error_values = [averageError, std_dev, ratioUnder*100, mse, mae]

    # Create a figure with a main plot and a subplot for the caption
    fig, (ax_main, ax_caption) = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 8))

    # Plot the results in the main subplot
    ax_main.plot(x_values, expected_values, label='Original Function', color='blue')
    ax_main.plot(x_values, y_values, label='FSM Approximation', linestyle='--', color='orange')
    ax_main.set_xlabel('x', fontsize=16)
    ax_main.set_ylabel('f(x)', fontsize=16)
    ax_main.set_title(f'FSM Approximation for {fsm.name}', fontsize=20)
    ax_main.legend(fontsize=16)
    ax_main.tick_params(axis='both', which='major', labelsize=14)  # Increase font size of axis ticks
    ax_main.grid(True)

    # Adding the figure caption in the caption subplot
    caption = ";   ".join([f"{label}: {value:.6f}" for label, value in zip(error_labels, error_values)])
    ax_caption.text(0.5, 0.5, caption, ha='center', va='center', fontsize=14)
    ax_caption.axis('off')  # Hide axis for the caption subplot

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


def generateStream(ratio, length):
    num_ones = int(ratio * length)
    num_zeros = length - num_ones

    # Create a list with the specified number of ones and zeros
    stream = ['1'] * num_ones + ['0'] * num_zeros

    # Shuffle the list to randomize the order of bits
    random.shuffle(stream)

    return stream


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


if __name__ == '__main__':
    #
    #   Determine FSM
    #
    salehiProposed = False
    ref65 = False
    roundingExample = False
    other = False
    bernstein = False
    approximations = True

    bipolarIssue = False
    subPath = "Reference"

    if salehiProposed:
        subPath = "Salehi"
        salehi_functon = "lambda x: (x**2 - 2*x + 1)/(x**2 - x + 1)"
        FSMSolver(salehi_functon, "Salehi Function", subPath=subPath, roundUp=False, logging=True)

    if ref65:
        subPath = "Ref65"
        function = "lambda x: (1/4) + (9 * x /8) - (15 * (x**2) / 8) + (5 * (x**3) / 4)"
        FSMSolver(function, "Ref65_Ex1", subPath=subPath, roundUp=False, logging=True)

        #bipolarIssue = True

    if roundingExample:
        subPath = "RoundingTest"
        function = "lambda x: (1/4) + (9 * x /8) - (15 * (x**2) / 8) + (5 * (x**3) / 4)"
        FSMSolver(function, "Ref65_Ex1", subPath=subPath, roundUp=False, logging=True)
        FSMSolver(function, "Ref65_Ex1 Rounded", subPath=subPath, roundUp=True, logging=True)

    if other:
        subPath = ""
        bipolarIssue = False

        quadratic = "lambda x: x**2"
        cubic = "lambda x: x**3"
        # FSMSolver(quadratic, "Quadratic", subPath=subPath, roundUp=False, logging=True)
        # FSMSolver(cubic, "Cubic", subPath=subPath, roundUp=False, logging=True)

        polynomial1 = "lambda x: (x**3) + (2*(x**2)) + (x/5) - (1/5)"  # random polynomial
        polynomial2 = "lambda x: (2 * (x**5) / 15) + ((x**3) / 3) + x"  # tan(x)
        polynomial3 = "lambda x: ((x**5) / 120) + ((x**4) / 24) + ((x**3) / 6) + ((x**2) / 2) + x"  # e^x - 1
        FSMSolver(polynomial1, "Random Polynomial", subPath=subPath, roundUp=False, logging=True)
        FSMSolver(polynomial2, "tan(x)", subPath=subPath, roundUp=False, logging=True)
        FSMSolver(polynomial3, "exp(x) - 1", subPath=subPath, roundUp=False, logging=True)

    if bernstein:
        subPath = "Bernstein"

        equation = "(x**3) + (2*(x**2)) + (x/5) - (1/5)"
        quadratic = lambda x: (x**3) + (2*(x**2)) + (x/5) - (1/5)
        bern_approx = Bernstein(quadratic, degree=10).construct_bernstein_polynomial()  # bernstein approx.

        FSMSolver(quadratic, "Random Polynomial", equation=equation, variable="x", subPath=subPath, roundUp=False, logging=True)
        FSMSolver(bern_approx, "Random Polynomial Bernstein Approximation", equation=equation, variable="x", subPath=subPath, roundUp=False, logging=True)

    if approximations:
        subPath = "Approximations"

        equation = "cos(x)"
        original = lambda x: cos(x)
        pade = lambda x: (1 - (115 * x**2) / 252 + (313 * x**4) / 15120) / (1 + (11 * x**2) / 252 + (13 * x**4) / 15120)
        maclaurin = lambda x: 1 - ((1/2) * x**2) + ((1/24) * x**4)
        bernstein_accurate = Bernstein(original, degree=4).construct_bernstein_polynomial()
        bernstein_expanded = Bernstein(original, degree=15).construct_bernstein_polynomial()

        FSMSolver(original, "Equation 3", degree=4, equation=equation, variable="x", subPath=subPath, roundUp=False, logging=True)
        FSMSolver(pade, "Equation 3 Pade-Type Approximation", equation=equation, variable="x", subPath=subPath, roundUp=False, logging=True)
        FSMSolver(maclaurin, "Equation 3 Maclaurin Approximation", equation=equation, variable="x", subPath=subPath, roundUp=False, logging=True)
        FSMSolver(bernstein_accurate, "Equation 3 Bernstein Approximation - 4th degree", equation=equation, variable="x", subPath=subPath, roundUp=False, logging=True)
        FSMSolver(bernstein_expanded, "Equation 3 Bernstein Approximation - 15th degree", equation=equation, variable="x", subPath=subPath, roundUp=False, logging=True)

    # Collect file names with .fsm extension
    fsm_files = glob.glob(f'FSMs/{subPath}/*.fsm')

    # fsm_files = [f"FSMs/{solver.fileStr}"]
    for test in fsm_files:
        print(f"Executing FSM: {test}")
        print('-' * 50)

        x_values = []
        expected_values = []
        y_values = []

        error = []

        func, fsm = parse_file(test)

        if not bipolarIssue:
            # Force Unipolar, until bipolar is fixed
            fsm.inputType = Types.UNIPOLAR
            fsm.outputType = Types.UNIPOLAR

        nsteps = 1000  # how many data points to collect
        lbound = -1 if fsm.inputType is Types.BIPOLAR else 0
        ubound = 1
        step = (ubound - lbound) / nsteps

        x_value = lbound

        while x_value <= ubound:
            expected = func(x_value)
            expected_values.append(expected)

            input_bits = generateStream(abs(x_value), 10000)
            input_fraction = np.count_nonzero(input_bits == '1') / len(input_bits)

            # Compute FSM approximation
            fsm_output = fsm.processInput(input_bits)
            y_value = np.sum(fsm_output) / len(fsm_output)  # f_p(x)

            if (fsm.quadrantIII and x_value < 0) or (fsm.quadrantIV and x_value > 1):
                if fsm.outputType == Types.BIPOLAR:
                    y_value = (2 * y_value) - 1  # compute f(x) from f_p(x)

            x_values.append(x_value)
            y_values.append(y_value)

            x_value += step

        table_values = [x_values, y_values, expected_values]
        print(tabulate(table_values, ["IN", "OUT", "f(x)"], tablefmt="pretty"))

        for i, y_value in enumerate(y_values):
            bottom_bound = 0 if fsm.outputType == Types.UNIPOLAR else 1

            if bottom_bound <= expected_values[i] <= 1:
                error.append(expected_values[i] - y_value)
            elif 1 <= expected_values[i]:
                error.append(1 - y_value)
            elif expected_values[i] <= bottom_bound:
                error.append(bottom_bound - y_value)

        fsm.error = error

        generatePlot(x_values, y_values, expected_values, fsm)
