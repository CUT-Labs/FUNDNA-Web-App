import sympy as sp
import numpy as np
from gui.classes.fsm.FunctionTypes import Types
from scipy.integrate import quad
from scipy.optimize import minimize
from tabulate import tabulate


class FSMSolver:
    def __init__(self, function, fsm_name, degree=None, equation=None, variable=None, subPath="", roundUp=False, logging=False, save=True):
        print("Initializing FSMSolver...")
        self.logging = logging
        self.name = fsm_name

        # string is an input such that:
        # lambda x: (x**2 - 2*x + 1)/(x**2 - x + 1)
        # or any other function

        if callable(function):
            assert equation is not None and variable is not None

            self.func = function
            if logging:
                print("Provided function is already callable!")
        else:
            self.func = eval(function)
            if logging:
                print("Provided function is NOT callable... function converted!")

        if equation is None or variable is None:
            self.variable = function.split(':')[0].split(' ')[1]
            self.original = function.split(': ')[1]
        else:
            self.original = equation
            self.variable = variable

        self.fileStr = f"{subPath}/{str.strip(fsm_name)}.fsm"

        self.quadrantI = False
        self.quadrantII = False
        self.quadrantIII = False
        self.quadrantIV = False

        self.numStates = self.numStates(degree=degree)  # number of states
        self.h_matrix = None
        self.c_vector = None
        self.b_vector = None
        self.states = None

        self.roundUp = roundUp

        print("FSMSolver Initialized.")
        print("Solving for output states...")
        self.solve()

        if save:
            self.toFile(roundUp=roundUp)

    def testFunctionType(self):
        def getYRange(func, xRange):
            yRange = []
            for x in xRange:
                yRange.append(func(x))

            return yRange

        x_range1 = np.linspace(-1, 0, 10)
        x_range2 = np.linspace(0, 1, 10)

        funcIn = Types.UNIPOLAR
        funcOut = Types.UNIPOLAR

        for y in getYRange(self.func, x_range1):  # negative x
            if -1 <= y <= 1:
                if -1 <= y < 0:
                    self.quadrantIII = True
                if 0 < y < 1:
                    self.quadrantII = True

        for y in getYRange(self.func, x_range2):  # positive x
            if -1 <= y <= 1:
                if -1 <= y < 0:
                    self.quadrantIV = True
                if 0 < y < 1:
                    self.quadrantI = True

        # if quadrantI is False and quadrantIV is False:
        #     raise SolverError(self, f"Input function not supported! No output in quadrant I or IV.")

        if self.quadrantII or self.quadrantIII:
            funcIn = Types.BIPOLAR

        if self.quadrantIII or self.quadrantIV:
            funcOut = Types.BIPOLAR

        return funcIn, funcOut

    def T(self, x):
        return self.func(x)

    def numStates(self, degree=None):
        assert degree is None or type(degree) is int

        if degree is None:
            x = sp.symbols('x')
            numerator = self.T(x).as_numer_denom()[0]  # Extract the numerator
            denominator = self.T(x).as_numer_denom()[1]  # Extract the denominator
            max_order = max(sp.degree(numerator, x), sp.degree(denominator, x))
        else:
            max_order = degree

        if self.logging:
            print('-' * 10)
            print(f"We need {max_order + 1} states for approximation.")
            print('-' * 10)

        return max_order + 1  # Add 1 to get the number of states

    # Define the objective function T(x)
    def objective_function(self, x):
        return 0.5 * np.dot(np.matrix.transpose(x), np.dot(self.h_matrix, x)) + np.dot(
            np.matrix.transpose(self.c_vector), x)

    def P_element(self, x):
        n = self.numStates
        p = [0] * n  # Initialize the array P_i with n elements, initially all zeros

        for i in range(n):
            numerator = (x / (1 - x)) ** i
            denominator = sum((x / (1 - x)) ** j for j in range(n))
            p[i] = numerator / denominator

        return p

    def P(self):
        n = self.numStates
        p = [None] * n  # Initialize the array to hold lambda functions

        for i in range(n):
            p[i] = lambda x, i=i, n=n: ((x / (1 - x)) ** i) / sum((x / (1 - x)) ** j for j in range(n))

        return p

    def P_approximate(self, test_point):
        p = self.P()
        results = []

        for i, P_i in enumerate(p):
            result = P_i(test_point)
            results.append([f'P_{i}({test_point})', result])

        if self.logging:
            print(tabulate(results, headers=['Function', 'Result'], tablefmt='grid'))

    def c_integrand(self, x, i):
        return -1 * self.T(x) * self.P_element(x)[i]

    def c_element(self, i):
        integral_result, _ = quad(self.c_integrand, 0, 1, args=(i))
        return integral_result

    def c(self):
        n = self.numStates
        c = np.zeros((n, 1))

        for i in range(n):
            c[i] = self.c_element(i)

        if self.logging:
            headers = ['']  # Column header for the table
            table_data = [[f'c_{i}'] + list(row) for i, row in enumerate(c)]

            print(tabulate(table_data, headers=headers, tablefmt='grid'))

        return c

    def H_integrand(self, x, i, j):
        return self.P_element(x)[i] * self.P_element(x)[j]

    def H_element(self, i, j):
        integral_result, _ = quad(self.H_integrand, 0, 1, args=(i, j))
        return integral_result

    def H(self):
        n = self.numStates
        h = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                h[i][j] = self.H_element(i, j)

        if self.logging:
            headers = [''] + [f'j = {j}' for j in range(n)]  # Column headers for the table
            table_data = [[f'i = {i}'] + list(row) for i, row in enumerate(h)]

            print(tabulate(table_data, headers=headers, tablefmt='grid'))

        return h

    def b(self):
        n = self.numStates
        h = self.h_matrix
        c = self.c_vector
        result = np.zeros((1, n))
        bounds = [(0, 1)] * n  # Assuming n states
        # Initial guess for x
        initial_guess = np.array([0.5 for i in range(n)])  # Or any other reasonable initial guess
        # Solve the optimization problem
        result = minimize(self.objective_function, initial_guess, bounds=bounds)

        if self.logging:
            # Convert optimal_x to a list for tabulation
            optimal_x_row_values = [[f's{i}', result.x[i]] for i in range(len(result.x))]
            print(tabulate(optimal_x_row_values, headers=['State', 'Output'], tablefmt='grid'))

        return result

    def solve(self):
        if self.logging:
            print("Begin solving FSM Approximation...")
            p_test = 0.5
            self.P_approximate(p_test)

        self.c_vector = self.c()
        self.h_matrix = self.H()
        self.b_vector = self.b().x

        # Get FSM State Information
        states = {}
        for i, val in enumerate(self.b_vector, 0):
            states[f"S{i}"] = round(float(val), 4) if not self.roundUp else round(float(val), 0)

        self.states = states

        return self.b_vector

    def toFile(self, roundUp=False):
        f = open(f"FSMs/{self.fileStr}", "w")

        # Determine FSM information to parse with FSMUtil
        f.write(f"Name: {self.name}\n")
        f.write(f"Variable: {self.variable}\n")
        f.write(f"Original: {self.original}\n")

        inputType, outputType = self.testFunctionType()

        f.write(f"Input: {'Bipolar' if inputType is Types.BIPOLAR else 'Unipolar'}\n")
        f.write(f"Output: {'Bipolar' if outputType is Types.BIPOLAR else 'Unipolar'}\n")

        # Get FSM State Information
        states = {}
        for i, val in enumerate(self.b_vector, 0):
            states[f"S{i}"] = round(float(val), 4) if not roundUp else round(float(val), 0)

        self.states = states
        f.write(f"Initial: {next(iter(states))}\n\n")

        # Write States to File
        f.write("States:\n")
        for state in states:
            f.write(f"  - {state}: {states[state]}\n")

        # Write Transitions to File
        f.write("\nTransitions:\n")
        for idx, state in enumerate(list(states.keys())):
            f.write(f"  - {state}:\n")  # Write state header
            # Handle transitions based on current state index
            if idx == 0:  # First state
                f.write(f"    0: {state}\n")  # Transition to self on 0
                f.write(f"    1: {list(states.keys())[idx + 1]}\n")  # Transition to next state on 1
            elif idx == len(states) - 1:  # Last state
                f.write(f"    0: {list(states.keys())[idx - 1]}\n")  # Transition to previous state on 0
                f.write(f"    1: {state}\n")  # Transition to self on 1
            else:  # Intermediate states
                f.write(f"    0: {list(states.keys())[idx - 1]}\n")  # Transition to previous state on 0
                f.write(f"    1: {list(states.keys())[idx + 1]}\n")  # Transition to next state on 1

        f.close()


class SolverError(Exception):
    def __init__(self, solver, message):
        self.message = message
        print(f"Solver {solver.name} has equation: {solver.original}")

    def __str__(self):
        return self.message
