from abc import ABC, abstractmethod
import numpy as np


class TranslationScheme(ABC):
    def __init__(self, states):
        self.crn = None
        self.rate_constants = None
        self.ode_strings = None
        self.ode_functions = None
        self.states = states
        self.num_states = len(states)
        self.reactions = []
        self.initial_conditions = {}

        self.create_translation_scheme()

        self.print_initial_conditions()

        self.get_crn()
        self.print_crn()

        self.generate_odes()

        self.print_odes()
        self.print_rate_constants()

    @abstractmethod
    def create_translation_scheme(self):
        """Define reactions and initial conditions specific to the scheme."""
        pass

    def get_crn(self):
        """Return the Chemical Reaction Network (CRN) based on the scheme."""
        return self.reactions

    def print_crn(self):
        """Print the Chemical Reaction Network (CRN) based on the scheme."""
        return self.reactions

    def get_initial_conditions(self):
        """Return the initial conditions for the states and outputs."""
        return self.initial_conditions

    def print_initial_conditions(self):
        """Print the initial conditions for the states and outputs."""
        print('\n')
        print('-' * 25)
        print("Initial Conditions:\n")
        for cond, value in self.initial_conditions.items():
            print(f"{cond} = {value}")

    def generate_odes(self):
        # Dictionary to store rate constant variables with initial guesses
        rate_constants = {}
        # List to store ODE functions
        ode_functions = []

        # Initialize the ODEs dictionary to construct expressions for each compound
        compounds = [f"S_{i}" for i in range(self.num_states)]
        compounds.append("Y_0")
        compounds.append("Y_1")

        ode_expressions = {f"S_{i}": "" for i in range(self.num_states)}
        ode_expressions["Y_0"] = ""
        ode_expressions["Y_1"] = ""

        # determine rate constants
        for i, reaction in enumerate(self.crn):
            rate_var = f"k_r{i}"
            rate_constants[rate_var] = reaction["rate"]

        # determine odes
        for c in compounds:
            positives = []
            negatives = []

            for i, reaction in enumerate(self.crn):
                if c in reaction["reactants"] and c not in reaction["products"]:
                    temp = f"k_r{i}"
                    for r in reaction["reactants"]:
                        temp += f" * {r}"
                    negatives.append(temp)
                if c in reaction["products"] and c not in reaction["reactants"]:
                    temp = f"k_r{i}"
                    for r in reaction["reactants"]:
                        temp += f" * {r}"
                    positives.append(temp)

            negStr = ' + '.join(negatives)
            posStr = ' + '.join(positives)

            exprStr = ""
            if posStr != "" and negStr != "":
                exprStr += f"({posStr}) - ({negStr})"
            else:
                if posStr != "":
                    exprStr += f"({posStr})"
                if negStr != "":
                    exprStr += f"-({negStr})"

            ode_expressions[c] = exprStr

        # Generate lambda functions for each ODE
        for compound, expr in ode_expressions.items():
            ode_functions.append(lambda y, t, k_dict=rate_constants, expr=expr: eval(expr, {}, {**y, **k_dict}))

        self.rate_constants = rate_constants
        self.ode_strings = ode_expressions
        self.ode_functions = ode_functions

        return rate_constants, ode_expressions, ode_functions

    def print_odes(self):
        """Print each ODE in a human-readable format."""
        print('\n')
        print('-' * 25)
        print("ODEs:\n")
        odes = self.ode_strings
        for compound, expr in odes.items():
            print(f"d{compound}/dt = {expr}")

    def print_rate_constants(self):
        """Print each rate constant and its initial value."""
        print('\n')
        print('-' * 25)
        print("Rate Constant Guesses:\n")
        for rate_name, value in self.rate_constants.items():
            print(f"{rate_name} = {value}")


class JPHuseProposedScheme(TranslationScheme):
    def create_translation_scheme(self):
        for i, value in enumerate(self.states):
            if i < self.num_states - 1:
                self.reactions.append((i, i + 1, value))
            self.initial_conditions[f"S_{i}"] = 0.01 if i == 0 else 0
        self.initial_conditions["Y_0"] = 0
        self.initial_conditions["Y_1"] = 0

    def get_crn(self):
        crn = []

        for i, state_output in enumerate(self.states):
            # Determine neighboring states
            left_state = i if i == 0 else i - 1
            right_state = i if i == (len(self.states) - 1) else i + 1

            left_output = self.states[left_state]
            right_output = self.states[right_state]

            # Four reactions for each state based on inputs 0 and 1
            crn.extend([
                {"reactants": [f"S_{i}", "X_0"], "products": [f"S_{left_state}", "Y_0"], "rate": 1 - left_output},
                {"reactants": [f"S_{i}", "X_0"], "products": [f"S_{left_state}", "Y_1"], "rate": left_output},
                {"reactants": [f"S_{i}", "X_1"], "products": [f"S_{right_state}", "Y_0"], "rate": 1 - right_output},
                {"reactants": [f"S_{i}", "X_1"], "products": [f"S_{right_state}", "Y_1"], "rate": right_output}
            ])

        self.crn = crn

        return crn

    def print_crn(self):
        # Retrieve the CRN to print it
        crn = self.get_crn()

        # Print each reaction in the specified format
        print('\n')
        print('-' * 25)
        print("CRN:\n")
        for reaction in crn:
            reactants_str = " + ".join(reaction["reactants"])
            products_str = " + ".join(reaction["products"])
            rate_str = f"k={reaction['rate']}"
            print(f"{reactants_str} -> {products_str}    {rate_str}")


# Example additional scheme (another placeholder for demonstration)
class AnotherScheme(TranslationScheme):
    def create_translation_scheme(self):
        # Define reactions specific to AnotherScheme
        pass

    def get_crn(self):
        # Define and return CRN for AnotherScheme
        pass


# Factory to easily add more schemes
def get_translation_scheme(scheme_name, states):
    schemes = {
        "JPHuseProposed": JPHuseProposedScheme,
        "AnotherScheme": AnotherScheme
        # Add additional schemes here
    }
    if scheme_name in schemes:
        return schemes[scheme_name](states)
    else:
        raise ValueError(f"Translation scheme '{scheme_name}' is not recognized.")
