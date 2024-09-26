from collections import defaultdict
from math import factorial

import matplotlib.pyplot as plt
import matplotlib.backends.backend_svg

import networkx as nx

import matplotlib
import matplotlib.pyplot as plt

from scipy.misc import derivative
from scipy.special import comb

import sympy as sp
from sympy.parsing.latex import parse_latex

from gui.classes.FuncTypes import FuncTypes
from gui.classes.GateTypes import GateTypes
from gui.classes.NotGateTypes import NotGateTypes

from gui.classes.Gate import Gate, PrintGateInfo

from gui.classes.CRN import *

import schemdraw
from schemdraw import logic


matplotlib.use('Agg')



def taylorToPolyStr(func, forceX):
    polynomial = ""
    if forceX:
        variable = "x"
    else:
        variable = func.variable

    for index in func.poli_coeffs:
        exponent = index
        coeff = round(float(func.poli_coeffs[index]), 4)

        if exponent == 0:
            polynomial = polynomial + str(coeff) + " + "
            continue

        if exponent == 1:
            polynomial = polynomial + str(coeff) + "*" + variable + " + "
            continue

        polynomial = polynomial + str(coeff) + "*" + variable + "^(" + str(exponent) + ") + "

    polynomial = polynomial[:-3]
    return polynomial


def doubleNANDFunctionToStr(func, forceX):
    doubleNand = ""
    coeffs = func.doubleNAND_coeffs
    if forceX:
        variable = "x"
    else:
        variable = func.variable

    if func.functype == FuncTypes.SINUSOIDAL:
        for index in coeffs:
            if not coeffs.keys().__contains__(0):
                if index % 2 == 0:  # even exponents
                    if index == list(coeffs.keys())[0]:  # outermost
                        doubleNand = doubleNand + variable + "^2*(1-" + str(round(coeffs[index], 4))
                    elif index == list(coeffs.keys())[-1]:  # innermost
                        doubleNand = doubleNand + "*(1-" + str(round(coeffs[index], 4)) + "*" + variable + "^2"
                    else:  # inner
                        doubleNand = doubleNand + "*(1-" + variable + "^2*(1-" + str(round(coeffs[index], 4))
                else:  # odd exponents
                    print("TODO FEATURE ERROR!!!")
    else:
        for index in coeffs:
            if index == list(coeffs.keys())[0]:  # outermost
                doubleNand = doubleNand + "1-" + str(round(coeffs[index], 4)) + "*("
            else:
                if index == list(coeffs.keys())[-1]:  # innermost
                    doubleNand = doubleNand + "1-" + str(round(coeffs[index], 4)) + "*" + variable
                else:  # inner
                    doubleNand = doubleNand + "1-" + variable + "*(1-" + str(round(coeffs[index], 4)) + "*("

    doubleNand = doubleNand + ")" * (doubleNand.count("("))
    return doubleNand


def hornerFunctionToStr(func, forceX):
    horner = ""
    coeffs = func.horner_coeffs
    if forceX:
        variable = "x"
    else:
        variable = func.variable

    if func.isSinusoidal():
        for index in coeffs:
            if index == 0:  # cos
                if 0.998 <= float(round(coeffs[index], 4)) <= 1.001:
                    continue
                horner = horner + str(round(coeffs[index], 4)) + "*("
            if index == 1:  # sin
                if 0.998 <= float(round(coeffs[index], 4)) <= 1.001:
                    horner = horner + variable + "*("
                    continue
                horner = horner + str(round(coeffs[index], 4)) + " * " + variable + "*("
            else:
                if 0.998 <= float(round(coeffs[index], 4)) <= 1.001:
                    horner = horner + "1-" + variable + "^2"
                else:
                    horner = horner + "1-" + str(round(coeffs[index], 4)) + "*" + variable + "^2"

                if list(coeffs.keys())[(len(coeffs) - 1)] != index:  # not last coeff, series continues
                    horner = horner + "*("
    else:
        for index in coeffs:
            if index == 0:
                if 0.998 <= float(round(coeffs[index], 4)) <= 1.001:
                    continue
                else:
                    horner = horner + str(round(coeffs[index], 4)) + "*("
            if index == 1:
                if 0.998 <= float(round(coeffs[index], 4)) <= 1.001:
                    horner = horner + "1-" + variable
                else:
                    horner = horner + "1-" + str(round(coeffs[index], 4)) + "*" + variable

                if list(coeffs.keys())[(len(coeffs) - 1)] != index:  # not last coeff, series continues
                    horner = horner + "*("
            if index != 1 and index != 0:
                if 0.998 <= float(round(coeffs[index], 4)) <= 1.001:
                    horner = horner + "1-" + variable
                else:
                    horner = horner + "1-" + str(round(coeffs[index], 4)) + "*" + variable

                if list(coeffs.keys())[(len(coeffs) - 1)] != index:  # not last coeff, series continues
                    horner = horner + "*("

    horner = horner + ")" * (horner.count("("))
    return horner


def make_taylor_coeffs(func, method="numerical"):
    """
    Generates the Maclaurin series (Taylor series at 0) coefficients for a given function.

    Args:
    - func: An internal function object that has 'function' (Python callable) and 'order' (the number of terms).
    - method: "numerical" (default) for numerical derivatives or "symbolic" for symbolic differentiation.

    Returns:
    - coeffs: A dictionary where the key is the term's order (n) and the value is the nth Taylor series coefficient.
    """
    coeffs = {}

    # Use SymPy for symbolic differentiation if method is set to "symbolic"
    if method == "symbolic":
        print("Using symbolic differentiation...")
        sympy_expr = parse_latex(func.latex)
        x = sp.symbols('x')  # Define the variable for SymPy

        for n in range(func.order):
            # Compute nth derivative using symbolic differentiation
            nth_derivative = sp.diff(sympy_expr, x, n)
            coeff = nth_derivative.subs(x, 0) / sp.factorial(n)  # Divide by n!
            coeffs[n] = float(coeff)  # Convert SymPy output to float
    else:
        print("Using numerical differentiation...")
        # Numerical differentiation using scipy.misc.derivative
        for n in range(func.order):
            # Ensure that `order` is always an odd number, even for small n values
            order = max(3, (n + 1) if n % 2 == 0 else (n + 2))

            # Calculate the nth derivative at 0 divided by n!
            try:
                coeff = derivative(func.function, 0, n=n, order=order, dx=1e-2) / factorial(n)
            except Exception as e:
                print(f"Error calculating numerical derivative for n={n}: {e}")
                coeff = 0.0  # Set to 0 in case of errors

            coeffs[n] = coeff

    return coeffs


def expand_binomial(point, n):
    coeffs = dict()
    for k in range(n + 1):
        coeff = ((-1) ** k) * comb(n, k) * (point ** k)
        coeffs[n - k] = coeff
    return coeffs


def make_polynomial(func, movePoint=False):
    if not movePoint:
        return func.taylor_coeffs
    else:
        new_taylor_coeffs = []
        final_dict = defaultdict(list)
        for power, coeff in func.taylor_coeffs.items():
            temp_dict = expand_binomial(func.point, power)
            temp_dict.update((x, y * coeff) for x, y in temp_dict.items())
            new_taylor_coeffs.append(temp_dict)
        for d in new_taylor_coeffs:
            for key, value in d.items():
                final_dict[key].append(value)
        final_dict.update((x, sum(y)) for x, y in final_dict.items())
        return dict(final_dict)


def ignore_small_coeffs(coeffs, ignore_th=1e-4):
    coeffs_new = {}
    for index in coeffs:
        if abs(coeffs[index]) > ignore_th:
            coeffs_new[index] = coeffs[index]
    return coeffs_new


def make_horner(func):
    func.horner_coeffs = ignore_small_coeffs(func.poli_coeffs)
    horner_coeffs = {}

    prev_index = 0
    for counter, index in enumerate(func.horner_coeffs):
        if counter == 0:
            horner_coeffs[index] = func.horner_coeffs[index]
        else:
            horner_coeffs[index] = -func.horner_coeffs[index] / func.horner_coeffs[prev_index]
        prev_index = index
    return horner_coeffs


def make_doubleNAND(func):
    func.doubleNAND_coeffs = ignore_small_coeffs(func.poli_coeffs)
    coeffs = {}
    tempList = list(func.doubleNAND_coeffs.values())
    tempListKeys = list(func.doubleNAND_coeffs.keys())

    for index in func.doubleNAND_coeffs:
        if index == 0:  # first coeff
            coeffs[index] = (1 - func.doubleNAND_coeffs[index])
        if index == tempListKeys[-1]:  # last in coeffs
            total = 0
            for indexj in func.doubleNAND_coeffs:
                total = total + func.doubleNAND_coeffs[indexj]
            total = total - tempList[-1]
            coeffs[index] = (func.doubleNAND_coeffs[index]) / (1 - total)
        else:
            total = 0
            for indexj in func.doubleNAND_coeffs:
                if index > indexj:
                    total = total + func.doubleNAND_coeffs[indexj]
            coeffs[index] = (1 - (total + func.doubleNAND_coeffs[index])) / (1 - total)
    return coeffs


def AddBaseGate(drawing, gateIndex, gateType, in1, in1Type, in2, in2Type, isXsquared):
    gateWrapper = Gate(gateType, in1, in1Type, in2, in2Type, None, None, gateIndex, True, isXsquared)

    if type(in1) is not str:
        in1 = str(round(in1, 4))

    if type(in2) is not str:
        in2 = str(round(in2, 4))

    with drawing as d:
        if gateType == GateTypes.NAND:
            d += (gate := logic.Nand().label(r'' + in1, 'in1').label(in2, 'in2')).right()
            d += gate.label("G" + str(gateIndex), 'center')
        if gateType == GateTypes.AND:
            d += (gate := logic.And().label(r'' + in1, 'in1').label(in2, 'in2')).right()
            d += gate.label("G" + str(gateIndex), 'center')

    gateWrapper.gate = gate

    return d, gateWrapper


def AddGateFromGate(drawing, gates, baseGate, gateType, gateIndex, inValue, inType, out, outType, connectBaseOut,
                    connectBaseIn2):
    gateWrapper = Gate(gateType, None, None, inValue, inType, out, outType, gateIndex, False, False)

    prevGate = gates[-1].gate
    base = baseGate.gate

    if type(inValue) is not str and inValue is not None:
        inValue = str(round(inValue, 4))

    with drawing as d:
        if not connectBaseOut and not connectBaseIn2:
            if gateType == GateTypes.NAND:
                d += (gate := logic.Nand().at(prevGate.out).anchor('in2').right().label(r'' + inValue, 'in1',
                                                                                        ofst=(0, 0.5)))
                d += gate.label("G" + str(gateIndex), 'center')
            if gateType == GateTypes.AND:
                d += (gate := logic.And().at(prevGate.out).anchor('in2').right().label(r'' + inValue, 'in1',
                                                                                       ofst=(0, 0.5)))
                d += gate.label("G" + str(gateIndex), 'center')
        if connectBaseIn2:
            if gateType == GateTypes.NAND:
                d += (gate := logic.Nand().at(prevGate.out).right().anchor('in1'))
                d += gate.label("G" + str(gateIndex), 'center')
            if gateType == GateTypes.AND:
                d += (gate := logic.And().at(prevGate.out).right().anchor('in1'))
                d += gate.label("G" + str(gateIndex), 'center')

            d += logic.Wire('n', k=-0.5).at(base.in2).to(gate.in2)
        if connectBaseOut:
            if gateType == GateTypes.NAND:
                d += (gate := logic.Nand().at(prevGate.out).right().anchor('in1'))
                d += gate.label("G" + str(gateIndex), 'center')
            if gateType == GateTypes.AND:
                d += (gate := logic.And().at(prevGate.out).right().anchor('in1'))
                d += gate.label("G" + str(gateIndex), 'center')

            d += logic.Wire('n', k=-0.5).at(base.out).to(gate.in2)

            for g in gates:
                if g.index is baseGate.index:
                    g.outputs.append("G" + str(gateIndex))
                    g.outputTypes.append(gateType)

    gateWrapper.input1 = "G" + str(gates[-1].index)
    gateWrapper.input1Type = gates[-1].gateType

    gates[-1].outputs.append("G" + str(gateIndex))
    gates[-1].outputTypes.append(gateType)

    if connectBaseOut or connectBaseIn2:
        for g in gates:
            if g.index is baseGate.index:
                gateWrapper.input2 = "G" + str(g.index)
                gateWrapper.input2Type = g.gateType

    gateWrapper.gate = gate

    gates.append(gateWrapper)

    return drawing, gates


def doubleNAND_to_circuit(func):
    gateIndex = 1
    gates = []
    variable = func.variable.upper()
    output = 'f(' + variable + ')'
    outputWithFormatting = r'$f(' + variable + ') = ' + func.title + '$'
    coeffs = dict(reversed(list(func.doubleNAND_coeffs.items())))

    schemdraw.use('matplotlib')  # Set backend

    try:
        print("Starting drawing...")
        # Initialize the drawing
        drawing = schemdraw.Drawing(show=False)  # Initialize the drawing manually
        print("Drawing started...")

        if func.isSinusoidal():  # uses x^2
            xSquaredGate = None
            for index in coeffs:
                print(f"Processing coefficient index: {index}")

                if index == list(coeffs.keys())[0]:  # First grouping (innermost)
                    # AND x with itself (x^2)
                    drawing, prevGate = AddBaseGate(drawing, gateIndex, GateTypes.AND,  # new gate info
                                                    variable, NotGateTypes.INPUT,  # in1 info
                                                    variable, NotGateTypes.INPUT, True)  # in2 info
                    gates.append(prevGate)
                    xSquaredGate = prevGate
                    gateIndex += 1

                    # NAND prev result with coeff
                    drawing, gates = AddGateFromGate(drawing, gates, gates[0], GateTypes.NAND, gateIndex,
                                                     coeffs[index], NotGateTypes.CONSTANT,
                                                     None, None, False, False)
                    gateIndex += 1

                elif index == list(coeffs.keys())[-1]:  # Outermost
                    # NAND prev result with coeff
                    drawing, gates = AddGateFromGate(drawing, gates, gates[0], GateTypes.NAND, gateIndex,
                                                     coeffs[index], NotGateTypes.CONSTANT,
                                                     None, None, False, False)
                    gateIndex += 1

                    # AND prev result with x^2, finish
                    drawing, gates = AddGateFromGate(drawing, gates, xSquaredGate, GateTypes.AND, gateIndex,
                                                     None, None,
                                                     None, None, True, False)
                    gateIndex += 1

                else:  # Inner groupings
                    # NAND prev result with coeff
                    drawing, gates = AddGateFromGate(drawing, gates, gates[0], GateTypes.NAND, gateIndex,
                                                     coeffs[index], NotGateTypes.CONSTANT,
                                                     None, None, False, False)
                    gateIndex += 1

                    # NAND prev result with x^2
                    drawing, gates = AddGateFromGate(drawing, gates, xSquaredGate, GateTypes.NAND, gateIndex,
                                                     None, None,
                                                     None, None, True, False)
                    gateIndex += 1

        else:  # Only uses x^1
            for index in coeffs:
                print(f"Processing coefficient index: {index}")

                if index == list(coeffs.keys())[0]:  # Innermost
                    # NAND coeff i==0 with x
                    drawing, prevGate = AddBaseGate(drawing, gateIndex, GateTypes.NAND,
                                                    coeffs[index], NotGateTypes.CONSTANT,
                                                    variable, NotGateTypes.INPUT,
                                                    False)
                    gates.append(prevGate)
                    gateIndex += 1

                elif index == list(coeffs.keys())[-1]:  # Outermost
                    # NAND coeff i==n with prev result, finish
                    drawing, gates = AddGateFromGate(drawing, gates, gates[0], GateTypes.NAND, gateIndex,
                                                     coeffs[index], NotGateTypes.CONSTANT,
                                                     None, None, False, False)
                    gateIndex += 1

                else:  # Middle section
                    # NAND coeff with prev result
                    drawing, gates = AddGateFromGate(drawing, gates, gates[0], GateTypes.NAND, gateIndex,
                                                     coeffs[index], NotGateTypes.CONSTANT,
                                                     None, None, False, False)
                    gateIndex += 1

                    # NAND x with prev result
                    drawing, gates = AddGateFromGate(drawing, gates, gates[0], GateTypes.NAND, gateIndex,
                                                     None, None,
                                                     None, None, False, True)
                    gateIndex += 1

        # Finalize circuit drawing
        drawing += gates[-1].gate.label(outputWithFormatting[0:5] + '$', 'out')
        print("...finished")

        svg_data = drawing.get_imagedata('svg')
        if not svg_data:
            raise ValueError("No SVG data returned from schemdraw.")

        print("SVG data retrieved successfully.")

        # Base64 encode SVG data
        import base64
        svg_base64 = base64.b64encode(svg_data).decode('utf-8')
        svg_url = f"data:image/svg+xml;base64,{svg_base64}"

        print(svg_url)
        print("Returning to function object")

        return svg_url, gates

    except Exception as e:
        print(f"An error occurred while generating the circuit: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def horner_to_circuit(func):
    gateIndex = 1
    gates = []
    variable = func.variable.upper()
    output = 'f(' + variable + ')'
    coeffs = func.horner_coeffs

    transCoeffs = reversed(coeffs)

    schemdraw.use('matplotlib')

    try:
        print("Starting drawing...")
        # Initialize the drawing
        drawing = schemdraw.Drawing(show=False)  # Initialize the drawing manually
        print("Drawing started...")

        if func.isSinusoidal():  # only uses x^2
            xSquaredGate = None
            for index in transCoeffs:
                print(f"Processing coefficient index: {index}")

                if list(coeffs.keys())[
                    (len(coeffs) - 1)] == index:  # First grouping (innermost 1-jx^2, where j is coeff)
                    # AND x with itself (x^2)
                    drawing, prevGate = AddBaseGate(drawing, gateIndex, GateTypes.AND,
                                                    variable, NotGateTypes.INPUT,
                                                    variable, NotGateTypes.INPUT, True)
                    xSquaredGate = prevGate
                    gates.append(prevGate)
                    gateIndex = gateIndex + 1

                    # NAND prev result with next coeff last coefficient
                    drawing, gates = AddGateFromGate(drawing, gates, gates[0], GateTypes.NAND, gateIndex,
                                                     coeffs[index], NotGateTypes.CONSTANT,
                                                     None, None, False, False)
                    gateIndex = gateIndex + 1

                else:
                    if index != 1 and index != 0:  # In between groupings (next few 1-jx^2, where j is coeff)
                        # AND prev result (gIndex - 2) with x^2 value
                        drawing, gates = AddGateFromGate(drawing, gates, xSquaredGate, GateTypes.AND, gateIndex,
                                                             None, None,
                                                             None, None, True, False)
                        gateIndex = gateIndex + 1

                        # NAND prev result with next coeff
                        drawing, gates = AddGateFromGate(drawing, gates, gates[0], GateTypes.NAND, gateIndex,
                                                             coeffs[index], NotGateTypes.CONSTANT,
                                                             None, None, False, False)
                        gateIndex = gateIndex + 1

                    else:  # last grouping where case 1: jx(1-kx^2(...)) is last group (i == 1), or case 2: 1-jx^2(...)
                        if index == 1:  # case 1, last term, sin(x)
                            # AND prev result with X
                            drawing, gates = AddGateFromGate(drawing, gates, gates[0], GateTypes.AND, gateIndex,
                                                                 None, None,
                                                                 None, None, False, True)
                            gateIndex = gateIndex + 1

                            # AND prev result with first coeff
                            if not FrivelousNumber(coeffs[index]):
                                drawing, gates = AddGateFromGate(drawing, gates, gates[0], GateTypes.AND, gateIndex,
                                                                     coeffs[index], NotGateTypes.CONSTANT,
                                                                     None, None, False, False)
                                gateIndex = gateIndex + 1

                        else:  # case 2, index == 0, last term, cos(x)
                            if not FrivelousNumber(coeffs[index]):
                                drawing, gates = AddGateFromGate(drawing, gates, gates[0], GateTypes.AND, gateIndex,
                                                                     coeffs[index], NotGateTypes.CONSTANT,
                                                                     None, None, False, False)
                                gateIndex = gateIndex + 1

        else:  # only uses x^1
            for index in transCoeffs:
                print(f"Processing coefficient index: {index}")

                if list(coeffs.keys())[
                    (len(coeffs) - 1)] == index:  # First grouping (innermost 1-jx, where j is coeff)
                    # NAND X and last coeff
                    drawing, prevGate = AddBaseGate(drawing, gateIndex, GateTypes.NAND,
                                                        coeffs[index], NotGateTypes.CONSTANT,
                                                        variable, NotGateTypes.INPUT, False)
                    gates.append(prevGate)
                    gateIndex = gateIndex + 1

                else:
                    if index != 1 and index != 0:  # In between groupings (next few 1-jx, where j is coeff)
                        if not FrivelousNumber(coeffs[index]):
                            drawing, gates = AddGateFromGate(drawing, gates, gates[0], GateTypes.AND, gateIndex,
                                                                 coeffs[index], NotGateTypes.CONSTANT,
                                                                 None, None, False, False)
                            gateIndex = gateIndex + 1

                        # NAND prev result with X
                        drawing, gates = AddGateFromGate(drawing, gates, gates[0], GateTypes.NAND, gateIndex,
                                                             None, None,
                                                             None, None, False, True)
                        gateIndex = gateIndex + 1

                    else:  # Last grouping, where case 1: jx(...) is last group (i == 1), or case 2: j(1-kx(...))
                        if 0 not in list(coeffs.keys()):  # case 1 - jx(...)
                            if not FrivelousNumber(coeffs[index]):
                                drawing, gates = AddGateFromGate(drawing, gates, gates[0], GateTypes.AND, gateIndex,
                                                                     coeffs[index], NotGateTypes.CONSTANT,
                                                                     None, None, False, False)
                                gateIndex = gateIndex + 1

                            drawing, gates = AddGateFromGate(drawing, gates, gates[0], GateTypes.AND, gateIndex,
                                                                 None, None,
                                                                 None, None, False, True)

                        else:  # case 2 - j(1-kx(...))
                            if index == 1:
                                if not FrivelousNumber(coeffs[index]):
                                    drawing, gates = AddGateFromGate(drawing, gates, gates[0], GateTypes.AND,
                                                                         gateIndex,
                                                                         coeffs[index], NotGateTypes.CONSTANT,
                                                                         None, None, False, False)
                                    prevGate = gates[-1]
                                    gateIndex = gateIndex + 1

                                drawing, gates = AddGateFromGate(drawing, gates, gates[0], GateTypes.NAND,
                                                                     gateIndex,
                                                                     None, None,
                                                                     None, None, False, True)
                                gateIndex = gateIndex + 1

                            else:  # index == 0
                                if not FrivelousNumber(coeffs[index]):
                                    drawing, gates = AddGateFromGate(drawing, gates, gates[0], GateTypes.AND,
                                                                         gateIndex,
                                                                         coeffs[index], NotGateTypes.CONSTANT,
                                                                         None, None, False, False)
                                    gateIndex = gateIndex + 1

        # Finalizing circuit drawing
        drawing += gates[-1].gate.label(output, 'out')
        print("...finished")

        svg_data = drawing.get_imagedata('svg')

        if not svg_data:
            raise ValueError("No SVG data returned from schemdraw.")

        print("SVG data retrieved successfully.")

        # Since `svg_data` is already in bytes, directly Base64 encode it
        import base64
        svg_base64 = base64.b64encode(svg_data).decode('utf-8')
        svg_url = f"data:image/svg+xml;base64,{svg_base64}"

        print(svg_url)

        print("Returning to function object")

        return svg_url, gates

    except Exception as e:
        print(f"An error occurred while generating the circuit: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def FrivelousNumber(number):
    if 0.998 <= number <= 1.001:
        return True
    else:
        return False


def show_circuit(func):
    func.circuit.save("assets/result.png")
    func.circuit.save("assets/result.svg")
    for gate in func.circuitGates:
        PrintGateInfo(gate)
    # func.circuit.draw()


def make_reactions(func):
    reactionStr = ""

    crn = CRN()

    for g in func.circuitGates:
        if GateTypes.isInEnum(g.gateType):
            gateType = g.gateType
            inputs = [g.input1, g.input2]
            outputs = g.outputs
            gateName = "G" + str(g.index)

            if type(inputs[0]) is not str and inputs[0] is not None:
                inputs[0] = str(round(inputs[0], 4))

            if type(inputs[1]) is not str and inputs[1] is not None:
                inputs[1] = str(round(inputs[1], 4))

            if gateType == GateTypes.AND:
                gateTypeStr = "M-AND"
            else:
                gateTypeStr = "M-NAND"

            reactionStr += gateName + "(" + gateTypeStr + ")\n"
            reactionStr += "Inputs: " + inputs[0] + ", " + inputs[1] + "\n"
            reactionStr += "Output(s) To: "

            for output in outputs:
                if output != outputs[0]:
                    reactionStr += ", " + output
                else:
                    reactionStr += " " + output

            reactionStr += "\n\nReaction Table:\n"

            gateReactionSet = make_reaction(gateType, inputs, gateName)

            for r in gateReactionSet:
                crn.AddReaction(r)
                print(r)
                reactionStr += r + "\n"
            print("-" * 100)

            reactionStr += "-" * 85 + "\n"

    print("\n\n\n\n\tREACTION STATEMENTS!!\n")
    print(reactionStr)
    return crn, reactionStr


def make_reaction(gateType, inputs, gateName):
    assert GateTypes.isInEnum(gateType)
    assert len(inputs) == 2

    print(gateType, "(", gateName, ")")
    print(inputs)

    a = inputs[0]
    b = inputs[1]
    c = gateName

    if not isinstance(a, str):
        a = str(round(a, 4))

    if not isinstance(b, str):
        b = str(round(b, 4))

    if not isinstance(c, str):
        c = str(c)

    if gateType == GateTypes.AND:
        reaction_list = [
            f"{a}_0 + {b}_0 -> {c}_0",
            f"{a}_0 + {b}_1 -> {c}_0",
            f"{a}_1 + {b}_0 -> {c}_0",
            f"{a}_1 + {b}_1 -> {c}_1",
        ]
    elif gateType == GateTypes.NAND:
        reaction_list = [
            f"{a}_0 + {b}_0 -> {c}_1",
            f"{a}_0 + {b}_1 -> {c}_1",
            f"{a}_1 + {b}_0 -> {c}_1",
            f"{a}_1 + {b}_1 -> {c}_0",
        ]
    else:
        print("GATE ERROR: Given Gate Type ", gateType)
        return -1

    return reaction_list
