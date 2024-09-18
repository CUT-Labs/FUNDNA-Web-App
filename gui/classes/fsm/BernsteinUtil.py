from math import *
import sympy as sp
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class Bernstein:
    def __init__(self, T, degree=10, logging=False):
        self.T = T  # T is the function to be approximated

        self.degree = degree
        self.logging = logging

        self.x = sp.symbols('x')  # Define x as a symbolic variable

    def numStates(self):
        numerator = self.T(self.x).as_numer_denom()[0]  # Extract the numerator
        denominator = self.T(self.x).as_numer_denom()[1]  # Extract the denominator
        max_order = max(sp.degree(numerator, self.x), sp.degree(denominator, self.x))

        if self.logging:
            print('-' * 10)
            print(f"We need {max_order + 1} states for approximation.")
            print('-' * 10)

        return max_order + 1  # Add 1 to get the number of states

    def binomial_coefficient(self, n, k):
        return sp.factorial(n) // (sp.factorial(k) * sp.factorial(n - k))

    def bernstein_polynomial_transform(self, n):
        b = [0] * (n + 1)

        for k in range(n + 1):
            b[k] = self.T(k / n) * self.binomial_coefficient(n, k) * (self.x ** k) * ((1 - self.x) ** (n - k))

        return b

    def construct_bernstein_polynomial(self):
        # degree = self.numStates() - 1
        b = self.bernstein_polynomial_transform(self.degree)
        print(b)

        poly_str = "lambda x: "

        for i, coeff in enumerate(b):
            if i != 0:
                poly_str += " + "
            poly_str += f"({coeff})"

        if self.logging:
            print(poly_str)

        return eval(poly_str)
