from enum import Enum


class GateTypes(Enum):
    NAND = "Nand"
    AND = "And"
    BNAND = "Bipolar Nand"
    BAND = "Bipolar And"
    MUX = "MUX"

    def isIn(gatetype):
        if type(gatetype) is str:
            if (gatetype == GateTypes.NAND.value or
                    gatetype == GateTypes.AND.value or
                    gatetype == GateTypes.BNAND.value or
                    gatetype == GateTypes.BAND.value or
                    gatetype == GateTypes.MUX.value):
                return 1
            else:
                return 0
        else:
            if (gatetype == GateTypes.NAND or
                    gatetype == GateTypes.AND or
                    gatetype == GateTypes.BNAND or
                    gatetype == GateTypes.BAND or
                    gatetype == GateTypes.MUX):
                return 1
            else:
                return 0

    def isInEnum(gatetype):
        if (gatetype == GateTypes.NAND or
                gatetype == GateTypes.AND or
                gatetype == GateTypes.BNAND or
                gatetype == GateTypes.BAND or
                gatetype == GateTypes.MUX):
            return 1
        else:
            return 0

    def isInValue(gatetype):
        if type(gatetype) is str:
            if (gatetype == GateTypes.NAND.value or
                    gatetype == GateTypes.AND.value or
                    gatetype == GateTypes.BNAND.value or
                    gatetype == GateTypes.BAND.value or
                    gatetype == GateTypes.MUX.value):
                return 1
        else:
            return 0
