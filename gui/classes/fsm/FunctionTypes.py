from enum import Enum

class Types(Enum):
    UNIPOLAR = "Unipolar"
    BIPOLAR = "Bipolar"

    def isIn(self):
        if type(self) is str:
            if self == Types.BIPOLAR.value or self == Types.UNIPOLAR.value:
                return 1
            else:
                return 0
        else:
            if self == Types.BIPOLAR or self == Types.UNIPOLAR:
                return 1
            else:
                return 0
