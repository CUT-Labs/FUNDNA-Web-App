from GateTypes import GateTypes
from NotGateTypes import NotGateTypes


def PrintGateInfo(gate):
    gate = CleanGateOutputs(gate)
    print('-' * 15)
    print("Gate G" + str(gate.index) + ": " + gate.gateType.value)
    print("isBase: " + str(gate.isBase))
    print("isXsquared: " + str(gate.isXsquared))

    if gate.input1Type is None:
        print("Input 1: " + str(gate.input1) + " (" + str(gate.input1Type) + ")")
    else:
        print("Input 1: " + str(gate.input1) + " (" + gate.input1Type.value + ")")

    if gate.input2Type is None:
        print("Input 2: " + str(gate.input2) + " (" + str(gate.input2Type) + ")")
    else:
        print("Input 2: " + str(gate.input2) + " (" + gate.input2Type.value + ")")

    if gate.outputTypes is None:
        print("Output: " + str(gate.outputs) + " (" + str(gate.outputTypes) + ")")
    else:
        print("Output: " + str(gate.outputs) + " (" + str(gate.outputTypes) + ")")


def CleanGateOutputs(gate):
    for out in gate.outputs:
        if out is None:
            gate.outputs.remove(out)
    for type in gate.outputTypes:
        if out is None:
            gate.outputTypes.remove(type)

    return gate


class Gate:
    def __init__(self, gateType, input1, input1Type, input2, input2Type, output, outputType, index, isBase, isXsquared):
        assert GateTypes.isInEnum(gateType)
        if input1 is not None:
            assert NotGateTypes.isInEnum(input1Type)
        if input2 is not None:
            assert NotGateTypes.isInEnum(input2Type)
        if output is not None:
            assert NotGateTypes.isInEnum(outputType)

        self.gate = None
        if isBase is None:
            self.isBase = True
        else:
            self.isBase = isBase

        if isXsquared is None:
            self.isXsquared = False
        else:
            self.isXsquared = isXsquared

        self.gateType = gateType
        self.input1 = input1
        self.input1Type = input1Type
        self.input2 = input2
        self.input2Type = input2Type
        self.outputs = [output]
        self.outputTypes = [outputType]
        self.index = index
