import numpy as np

class Sequence:
    def __init__(self, name, seq):
        self.Name = name
        self.Sequence = seq


class Strand:
    def __init__(self, name, strand, issignal):
        self.Name = name
        self.Strand = strand
        self.IsSignal = issignal


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

    def Min(self, weighted=False):
        min_score = None
        min_test_name = None
        for test_name, (score, weight) in self.ToDict().items():
            if score is None:
                continue
            # Apply weighting if specified
            score_to_check = score * weight / 100 if weighted else score
            if min_score is None or score_to_check < min_score:
                min_score = score_to_check
                min_test_name = test_name
        return min_test_name, min_score

    def Max(self, weighted=False):
        max_score = None
        max_test_name = None
        for test_name, (score, weight) in self.ToDict().items():
            if score is None:
                continue
            # Apply weighting if specified
            score_to_check = score * weight / 100 if weighted else score
            if max_score is None or score_to_check > max_score:
                max_score = score_to_check
                max_test_name = test_name
        return max_test_name, max_score


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

    def MetaRanksArray(self):
        meta_scores = {
            "Design": [],
            "Worst-rank": [],
            "Worst-weighted-rank": [],
            "Sum-of-ranks": [],
            "Weighted Sum-of-ranks": [],
            "Fractional-excess": [],
            "Weighted Fractional-excess": [],
            "Percent-badness": [],
            "Weighted Percent-badness": []
        }

        for design in self.Designs:
            # Calculate scores for each category
            worst_rank = design.RankArray.Max()[1]
            worst_weighted_rank = design.RankArray.Max(weighted=True)[1]
            sum_of_ranks = design.RankArray.Sum()
            weighted_sum_of_ranks = design.RankArray.WeightedSum()
            fractional_excess = design.FractionalExcessArray.Sum()
            weighted_fractional_excess = design.FractionalExcessArray.WeightedSum()
            percent_badness = design.PercentBadnessArray.Sum()
            weighted_percent_badness = design.PercentBadnessArray.WeightedSum()

            # Collect scores for ranking
            meta_scores["Design"].append(design.Name)
            meta_scores["Worst-rank"].append(worst_rank)
            meta_scores["Worst-weighted-rank"].append(worst_weighted_rank)
            meta_scores["Sum-of-ranks"].append(sum_of_ranks)
            meta_scores["Weighted Sum-of-ranks"].append(weighted_sum_of_ranks)
            meta_scores["Fractional-excess"].append(fractional_excess)
            meta_scores["Weighted Fractional-excess"].append(weighted_fractional_excess)
            meta_scores["Percent-badness"].append(percent_badness)
            meta_scores["Weighted Percent-badness"].append(weighted_percent_badness)

        # Rank each design based on meta scores and add Meta Sum to ranked_meta_scores
        ranked_meta_scores = {"Design": meta_scores["Design"]}
        for category, scores in meta_scores.items():
            if category != "Design":
                ranked_meta_scores[category] = self.rank_values(scores)

        # Calculate Meta Sum for each design
        meta_ranks_sum = [
            sum(ranked_meta_scores[category][i] for category in ranked_meta_scores if category != "Design")
            for i in range(len(self.Designs))
        ]
        ranked_meta_scores["Meta Sum"] = meta_ranks_sum

        # Determine the best Meta Sum and best designs based on Meta Sum
        best_meta_sum = min(meta_ranks_sum)
        best_designs = [self.Designs[i].Name for i, sum_rank in enumerate(meta_ranks_sum) if sum_rank == best_meta_sum]

        # Debug output
        print("Meta Scores:")
        print(meta_scores)
        print("Ranked Meta Scores:")
        print(ranked_meta_scores)
        print("Best Meta Sum:")
        print(best_meta_sum)
        print("Best Designs:")
        print(best_designs)

        return meta_scores, ranked_meta_scores, best_meta_sum, best_designs

    def WorstRank(self):
        scores = [design.RankArray.Max()[1] for design in self.Designs]
        best_score = min(scores)
        worst_score = max(scores)
        best_design = [design.Name for design, score in zip(self.Designs, scores) if score == best_score]
        worst_design = [design.Name for design, score in zip(self.Designs, scores) if score == worst_score]
        return (best_score, best_design), (worst_score, worst_design)

    def WorstWeightedRank(self):
        scores = [design.RankArray.Max(weighted=True)[1] for design in self.Designs]
        best_score = min(scores)
        worst_score = max(scores)
        best_design = [design.Name for design, score in zip(self.Designs, scores) if score == best_score]
        worst_design = [design.Name for design, score in zip(self.Designs, scores) if score == worst_score]
        return (best_score, best_design), (worst_score, worst_design)

    def SumOfRanks(self):
        scores = [design.RankArray.Sum() for design in self.Designs]
        best_score = min(scores)
        worst_score = max(scores)
        best_design = [design.Name for design, score in zip(self.Designs, scores) if score == best_score]
        worst_design = [design.Name for design, score in zip(self.Designs, scores) if score == worst_score]
        return (best_score, best_design), (worst_score, worst_design)

    def WeightedSumOfRank(self):
        scores = [design.RankArray.WeightedSum() for design in self.Designs]
        best_score = min(scores)
        worst_score = max(scores)
        best_design = [design.Name for design, score in zip(self.Designs, scores) if score == best_score]
        worst_design = [design.Name for design, score in zip(self.Designs, scores) if score == worst_score]
        return (best_score, best_design), (worst_score, worst_design)

    def FractionalExcess(self):
        scores = [design.FractionalExcessArray.Sum() for design in self.Designs]
        best_score = min(scores)
        worst_score = max(scores)
        best_design = [design.Name for design, score in zip(self.Designs, scores) if score == best_score]
        worst_design = [design.Name for design, score in zip(self.Designs, scores) if score == worst_score]
        return (best_score, best_design), (worst_score, worst_design)

    def WeightedFractionalExcess(self):
        scores = [design.FractionalExcessArray.WeightedSum() for design in self.Designs]
        best_score = min(scores)
        worst_score = max(scores)
        best_design = [design.Name for design, score in zip(self.Designs, scores) if score == best_score]
        worst_design = [design.Name for design, score in zip(self.Designs, scores) if score == worst_score]
        return (best_score, best_design), (worst_score, worst_design)

    def PercentBadness(self):
        scores = [design.PercentBadnessArray.Sum() for design in self.Designs]
        best_score = min(scores)
        worst_score = max(scores)
        best_design = [design.Name for design, score in zip(self.Designs, scores) if score == best_score]
        worst_design = [design.Name for design, score in zip(self.Designs, scores) if score == worst_score]
        return (best_score, best_design), (worst_score, worst_design)

    def WeightedPercentBadness(self):
        scores = [design.PercentBadnessArray.WeightedSum() for design in self.Designs]
        best_score = min(scores)
        worst_score = max(scores)
        best_design = [design.Name for design, score in zip(self.Designs, scores) if score == best_score]
        worst_design = [design.Name for design, score in zip(self.Designs, scores) if score == worst_score]
        return (best_score, best_design), (worst_score, worst_design)
