from enum import Enum


class NotGateTypes(Enum):
    INPUT = "Input"
    OUTPUT = "Output"
    CONSTANT = "Const"
    INTERMEDIATE = "Intermediate"

    def isIn(functype):
        if type(functype) is str:
            if (functype == NotGateTypes.INPUT.value or
                    functype == NotGateTypes.OUTPUT.value or
                    functype == NotGateTypes.CONSTANT.value or
                    functype == NotGateTypes.INTERMEDIATE.value):
                return 1
            else:
                return 0
        else:
            if (functype == NotGateTypes.INPUT or
                    functype == NotGateTypes.OUTPUT or
                    functype == NotGateTypes.CONSTANT or
                    functype == NotGateTypes.INTERMEDIATE):
                return 1
            else:
                return 0

    def isInEnum(functype):
        if (functype == NotGateTypes.INPUT or
                functype == NotGateTypes.OUTPUT or
                functype == NotGateTypes.CONSTANT or
                functype == NotGateTypes.INTERMEDIATE):
            return 1
        else:
            return 0

    def isInValue(functype):
        if type(functype) is str:
            if (functype == NotGateTypes.INPUT.value or
                    functype == NotGateTypes.OUTPUT.value or
                    functype == NotGateTypes.CONSTANT.value or
                    functype == NotGateTypes.INTERMEDIATE.value):
                return 1
        else:
            return 0
