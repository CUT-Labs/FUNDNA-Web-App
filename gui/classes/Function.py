from Util import *
from FuncTypes import FuncTypes
from RearrangeType import RearrangeType


class Function:
    def __init__(self, function, point, order, functype, title, variable):
        assert FuncTypes.isIn(functype)

        self.circuit = None
        self.circuitGates = None
        self.horner_coeffs = None
        self.doubleNAND_coeffs = None
        self.poli_coeffs = None
        self.taylor_coeffs = None
        self.taylorString = None
        self.rearrangeString = None
        self.CRN = None
        self.GUIReactionTable = None
        self.traceString = None
        self.traceValue = 0.0
        self.rearrangeType = None
        self.function = function
        self.point = point
        self.order = order
        self.functype = functype
        self.title = title
        self.variable = variable

    def generateCoeffs(self):
        print("-" * 100)
        print("Taylor Coeffs")
        print("")
        self.taylor_coeffs = make_taylor_coeffs(self)
        for index in self.taylor_coeffs:
            print(index, ": ", self.taylor_coeffs[index])

        print("-" * 100)
        print("Polynomial Coeffs")
        print("")
        self.poli_coeffs = make_polynomial(self)
        for index in self.poli_coeffs:
            print(index, ": ", self.poli_coeffs[index])

        self.taylorString = taylorToPolyStr(self, 0)
        print("Taylor Polynomial String: " + self.taylorString)

        self.rearrangeType = self.determineRearrangement()

        if self.rearrangeType == RearrangeType.DOUBLE_NAND:
            print("-" * 100)
            print("Double NAND Expansion Coeffs")
            print("")
            self.doubleNAND_coeffs = make_doubleNAND(self)
            for index in self.doubleNAND_coeffs:
                print(index, ": ", self.doubleNAND_coeffs[index])
            self.rearrangeString = doubleNANDFunctionToStr(self, 0)
            print("Rearrangement String: " + self.rearrangeString)
        if self.rearrangeType == RearrangeType.HORNER:
            print("-" * 100)
            print("Horner Expansion Coeffs")
            print("")
            self.horner_coeffs = make_horner(self)
            for index in self.horner_coeffs:
                print(index, ": ", self.horner_coeffs[index])
            self.rearrangeString = hornerFunctionToStr(self, 0)
            print("Rearrangement String" + self.rearrangeString)

    def determineRearrangement(self):
        # HORNER:      alternating signs, coeff decreases as power increases
        # DOUBLE NAND: 0 <= coeffs <= infinity, 0 <= sum <= 1
        fixedPoliCoeffs = ignore_small_coeffs(self.poli_coeffs)
        alternatingSign = -1
        decreasingCoeffs = 1
        allPositive = 1
        poli_total = 0.0

        print("-" * 100)

        # total coefficients
        for index in fixedPoliCoeffs:
            poli_total = poli_total + fixedPoliCoeffs[index]

        # determine if coeffs are all positive
        for index in fixedPoliCoeffs:
            if allPositive == 0:
                break

            if fixedPoliCoeffs[index] > 0 or fixedPoliCoeffs[index] == 0:
                allPositive = 1
            else:
                allPositive = 0

        # Determine if decreasing coefficients
        lastValue = 0
        for index in fixedPoliCoeffs:
            if decreasingCoeffs == 0:
                break

            if fixedPoliCoeffs[index] == 0:
                continue

            if lastValue == 0:
                lastValue = round(abs(fixedPoliCoeffs[index]), 4)
                continue

            if abs(lastValue) < round(abs(fixedPoliCoeffs[index]), 4):
                decreasingCoeffs = 0
                continue

            if abs(lastValue) > round(abs(fixedPoliCoeffs[index]), 4):
                lastValue = round(abs(fixedPoliCoeffs[index]), 4)
                continue

            if abs(lastValue) == round(abs(fixedPoliCoeffs[index]), 4):
                continue

        # Determine if Alternating Signs
        if allPositive == 0:
            lastValue = 0
            for index in fixedPoliCoeffs:
                if alternatingSign == 0:
                    break
                if fixedPoliCoeffs[index] != 0:
                    if fixedPoliCoeffs[index] < 0:
                        print(str(fixedPoliCoeffs[index]) + " is NEG")
                        if lastValue == -1:
                            alternatingSign = 0
                        else:
                            lastValue = -1
                    elif self.poli_coeffs[index] > 0:
                        print(str(fixedPoliCoeffs[index]) + " is POS")
                        if lastValue == 1:
                            alternatingSign = 0
                        else:
                            lastValue = 1
                    else:
                        print(str(fixedPoliCoeffs[index]) + " is ZERO, IGNORING")
                        continue
            if alternatingSign != 0:
                alternatingSign = 1

        print("\nAlternating: " + alternatingSign.__str__())
        print("Decreasing: " + decreasingCoeffs.__str__())
        print("Coeff Sum: " + poli_total.__str__())
        print("All Positive: " + allPositive.__str__())

        if 0 <= poli_total <= 1 and allPositive == 1:
            print("Rearrangement Type => Double NAND Replacement")
            print("-" * 100)
            return RearrangeType.DOUBLE_NAND
        if alternatingSign == 1 and decreasingCoeffs == 1:
            print("Rearrangement Type => Horner")
            print("-" * 100)
            return RearrangeType.HORNER
        else:
            print("Rearrangement Type => Unknown")
            print("-" * 100)
            return RearrangeType.UNKNOWN

    def generateCircuit(self):
        if self.rearrangeType == RearrangeType.DOUBLE_NAND:
            self.circuit, self.circuitGates = doubleNAND_to_circuit(self)
        if self.rearrangeType == RearrangeType.HORNER:
            self.circuit, self.circuitGates = horner_to_circuit(self)

        self.circuit.save('assets/result.svg')

        print("Circuit returned to function!")

        if self.circuit is None and self.circuitGates is None:
            raise Exception("so actually circuit drawing nor gates made it back to the function object")
        elif self.circuit is None:
            raise Exception("idk what's happening, but the circuit drawing didn't make it back to the function object")
        elif self.circuitGates is None:
            raise Exception("idk what's happening, but the gates didn't make it back to the function object")

        if self.rearrangeType != RearrangeType.UNKNOWN:
            show_circuit(self)

    def generateReactions(self):
        if self.rearrangeType != RearrangeType.UNKNOWN:
            self.CRN, self.GUIReactionTable = make_reactions(self)

    def generateTrace(self):
        if self.rearrangeType == RearrangeType.DOUBLE_NAND:
            self.traceString = doubleNANDFunctionToStr(self, 1)
        if self.rearrangeType == RearrangeType.HORNER:
            self.traceString = hornerFunctionToStr(self, 1)

        x = self.point
        self.traceValue = eval(self.traceString.replace("^", "**"))

    def isSinusoidal(self):
        if self.functype == FuncTypes.SINUSOIDAL:
            return 1
        elif self.functype == FuncTypes.SINE:
            return 1
        elif self.functype == FuncTypes.COSINE:
            return 1
        else:
            return 0

    def isExponential(self):
        if self.functype == FuncTypes.EXPONENTIAL:
            return 1
        else:
            return 0

    def isLogarithmic(self):
        if self.functype == FuncTypes.LOGARITHMIC:
            return 1
        else:
            return 0

    def isPolynomial(self):
        if self.functype == FuncTypes.POLYNOMIAL:
            return 1
        else:
            return 0

    # def generateNuskellString(self):
    #     nuskellCRN = ""
    #     for g in self.circuitGates:
    #         if GateTypes.isInEnum(g.gateType):
    #             gateType = g.gateType
    #             inputs = [g.input1, g.input2]
    #             gateName = "G" + str(g.index)
    #
    #             if type(inputs[0]) is not str and inputs[0] is not None:
    #                 inputs[0] = str(round(inputs[0], 4))
    #
    #             if type(inputs[1]) is not str and inputs[1] is not None:
    #                 inputs[1] = str(round(inputs[1], 4))
    #
    #             reaction = make_reaction(gateType, inputs, gateName)
    #
    #             nuskellCRN += reaction + "; "
    #
    #     return nuskellCRN.rstrip("; ")  # Remove the trailing semicolon
    #
    # def generatePiperineString(self):
    #     piperineCRN = ""
    #     for g in self.circuitGates:
    #         if GateTypes.isInEnum(g.gateType):
    #             gateType = g.gateType
    #             inputs = [g.input1, g.input2]
    #             gateName = "G" + str(g.index)
    #
    #             if type(inputs[0]) is not str and inputs[0] is not None:
    #                 inputs[0] = str(round(inputs[0], 4))
    #
    #             if type(inputs[1]) is not str and inputs[1] is not None:
    #                 inputs[1] = str(round(inputs[1], 4))
    #
    #             reaction = make_reaction(gateType, inputs, gateName)
    #
    #             piperineCRN += reaction.replace("; ", "\n") + "\n"
    #
    #     return piperineCRN.rstrip("\n")  # Remove the trailing semicolon
