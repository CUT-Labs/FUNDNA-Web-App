from enum import Enum


class FuncTypes(Enum):
    SINUSOIDAL = "Sinusoidal"
    SINE = "Sin"
    COSINE = "Cos"
    EXPONENTIAL = "Exponential"
    LOGARITHMIC = "Log"
    POLYNOMIAL = "Polynomial"

    def isIn(self):
        if type(self) is str:
            if (self == FuncTypes.SINUSOIDAL.value or
                    self == FuncTypes.SINE.value or
                    self == FuncTypes.COSINE.value or
                    self == FuncTypes.EXPONENTIAL.value or
                    self == FuncTypes.LOGARITHMIC.value or
                    self == FuncTypes.POLYNOMIAL.value):
                return 1
            else:
                return 0
        else:
            if (self == FuncTypes.SINUSOIDAL or
                    self == FuncTypes.SINE or
                    self == FuncTypes.COSINE or
                    self == FuncTypes.EXPONENTIAL or
                    self == FuncTypes.LOGARITHMIC or
                    self == FuncTypes.POLYNOMIAL):
                return 1
            else:
                return 0