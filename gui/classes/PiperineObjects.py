class Sequence:
    def __init__(self, name, seq):
        self.Name = name
        self.Sequence = seq


class Strand:
    def __init__(self, name, strand):
        self.Name = name
        self.Strand = strand


class Structure:
    def __init__(self, name, struct):
        self.Name = name
        self.Structure = struct


class Complex:
    def __init__(self, name, strands, isfuel):
        self.Name = name
        self.Strands = strands
        self.IsFuel = isfuel


class Design:
    def __init__(self, name):
        self.Name = name
        self.SignalStrands = []  # from my[i]_strands.txt
        self.Complexes = []  # from my[i]_strands.txt
        self.Sequences = []  # from my[i].seqs
        self.Strands = []  # from my[i].seqs
        self.Structures = []  # from my[i].seqs

        self.RawScores = ScoresArray()
        self.RankArray = ScoresArray()
        self.FractionalExcessArray = ScoresArray()
        self.PercentBadnessArray = ScoresArray()


class ScoresArray:
    def __init__(self):
        self.TSI_avg = None
        self.TSI_avg_weight = 5
        self.TSI_max = None
        self.TSI_max_weight = 20
        self.TO_avg = None
        self.TO_avg_weight = 10
        self.TO_max = None
        self.TO_max_weight = 30
        self.WS_BM = None
        self.WS_BM_weight = 2
        self.Max_BM = None
        self.Max_BM_weight = 3
        self.SSU_min = None
        self.SSU_min_weight = 30
        self.SSU_avg = None
        self.SSU_avg_weight = 10
        self.SSTU_min = None
        self.SSTU_min_weight = 50
        self.SSTU_avg = None
        self.SSTU_avg_weight = 20
        self.BN_percent_max = None
        self.BN_percent_max_weight = 10
        self.BN_percent_avg = None
        self.BN_percent_avg_weight = 5
        self.WSAS = None
        self.WSAS_weight = 6
        self.WSIS = None
        self.WSIS_weight = 4
        self.WSAS_M = None
        self.WSAS_M_weight = 5
        self.WSIS_M = None
        self.WSIS_M_weight = 3
        self.Verboten = None
        self.Verboten_weight = 2
        self.Spurious = None
        self.Spurious_weight = 8
        self.dG_Error = None
        self.dG_Error_weight = 10
        self.dG_Range = None
        self.dG_Range_weight = 20

    def ToDict(self):
        return {
            "TSI avg": (self.TSI_avg, self.TSI_avg_weight),
            "TSI max": (self.TSI_max, self.TSI_max_weight),
            "TO avg": (self.TO_avg, self.TO_avg_weight),
            "TO max": (self.TO_max, self.TO_max_weight),
            "WS-BM": (self.WS_BM, self.WS_BM_weight),
            "Max-BM": (self.Max_BM, self.Max_BM_weight),
            "SSU min": (self.SSU_min, self.SSU_min_weight),
            "SSU avg": (self.SSU_avg, self.SSU_avg_weight),
            "SSTU min": (self.SSTU_min, self.SSTU_min_weight),
            "SSTU avg": (self.SSTU_avg, self.SSTU_avg_weight),
            "BN% max": (self.BN_percent_max, self.BN_percent_max_weight),
            "BN% avg": (self.BN_percent_avg, self.BN_percent_avg_weight),
            "WSAS": (self.WSAS, self.WSAS_weight),
            "WSIS": (self.WSIS, self.WSIS_weight),
            "WSAS-M": (self.WSAS_M, self.WSAS_M_weight),
            "WSIS-M": (self.WSIS_M, self.WSIS_M_weight),
            "Verboten": (self.Verboten, self.Verboten_weight),
            "Spurious": (self.Spurious, self.Spurious_weight),
            "dG Error": (self.dG_Error, self.dG_Error_weight),
            "dG Range": (self.dG_Range, self.dG_Range_weight)
        }

    def from_list(self, values_list):
        keys = [
            "TSI_avg", "TSI_max", "TO_avg", "TO_max", "WS_BM", "Max_BM",
            "SSU_min", "SSU_avg", "SSTU_min", "SSTU_avg", "BN_percent_max",
            "BN_percent_avg", "WSAS", "WSIS", "WSAS_M", "WSIS_M",
            "Verboten", "Spurious", "dG_Error", "dG_Range"
        ]
        for key, value in zip(keys, values_list):
            setattr(self, key, value)

    def Sum(self):
        # Sum of all scores that are not None
        total = 0
        for key, value in self.ToDict().items():
            score, _ = value
            if score is not None:
                total += score
        return total

    def WeightedSum(self):
        # Weighted sum of all scores that are not None
        total = 0
        for key, value in self.ToDict().items():
            score, weight = value
            if score is not None:
                total += score * (weight / 100)
        return total


class PiperineOutput:
    def __init__(self):
        self.Designs = []

    def rank_values(self, values, reversed=False):
        # Rank the values, lower is better by default unless reversed is True
        array = np.array(values)
        if reversed:
            array = -array
        unique_values = np.unique(array)
        rank_dict = {val: rank for rank, val in enumerate(unique_values)}
        return [rank_dict[v] for v in array]

    # Worst-rank
    # (weighted)
    # Sum-of-ranks
    # (weighted)
    # Fractional-excess
    # (weighted)
    # Percent-badness
    # (weighted)
    def MetaRanksArray(self):
        rank_arrays = [design.RankArray.ToDict() for design in self.Designs]
