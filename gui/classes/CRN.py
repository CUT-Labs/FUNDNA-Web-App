class CRN:
    def __init__(self, crn_array=None, file_reference=None):
        # Initialize the CRN with an array of reactions and a file reference
        self.crn_array = crn_array if crn_array else []
        self.file_reference = file_reference
        self.gates = set()  # Set to store distinct gates

    def AddReaction(self, reaction):
        assert type(reaction) is str
        self.crn_array.append(reaction)

    def CountSpecies(self):
        # Count each distinct gate in the entire CRN
        for reaction in self.crn_array:
            reactants, products = reaction.split("->")
            all_gates = reactants.split() + products.split()
            self.gates.update(all_gates)
        return len(self.gates)

    def HasFile(self):
        if self.file_reference:
            return True
        else:
            return False

    def OpenFile(self):
        # Open a file only if there is no file reference
        if not self.file_reference:
            self.file_reference = open("tests/my.crn", "w+")

    def CloseFile(self):
        # Close the file only if there is a file reference
        if self.file_reference:
            self.file_reference.close()
            print("File closed.")
        else:
            print("No file to close.")

    def NuskellString(self):
        nuskellStr = ""

        for reaction in self.crn_array:
            nuskellStr += reaction
            nuskellStr += "; "

        return nuskellStr.rstrip("; ")  # Remove the trailing semicolon

    def PiperineString(self):
        piperineStr = ""

        for reaction in self.crn_array:
            piperineStr += reaction
            piperineStr += "\n"

        return piperineStr.rstrip("\n")  # Remove the trailing new line
