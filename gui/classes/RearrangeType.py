from enum import Enum


class RearrangeType(Enum):
    HORNER = "Horner"
    DOUBLE_NAND = "Double NAND"
    UNKNOWN = "Unknown"

    def isIn(self):
        if type(self) is str:
            if (self == RearrangeType.HORNER.value or
                    self == RearrangeType.DOUBLE_NAND.value or
                    self == RearrangeType.UNKNOWN.value):
                return 1
            else:
                return 0
        else:
            if (self == RearrangeType.HORNER or
                    self == RearrangeType.DOUBLE_NAND or
                    self == RearrangeType.UNKNOWN):
                return 1
            else:
                return 0
