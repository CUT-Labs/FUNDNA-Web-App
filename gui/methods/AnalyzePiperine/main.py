"""

----------------------------------------------

How to determine a structure's shape

structure r0-Trans = GATGAGAAGTTAAGTGGAAGAGAAAGGATTGTTGAGTAGTATTGAATGAAGGT+TCTACACTTCTTATCCCCCAACCTTCATTCAATACTACTCAACA+AAAACACTCATAAAATTTTCTCATCCTTTCTCTTCCACTTAACT

Step 1. Separate
A: 5' GATGAGAAGTTAAGTGGAAGAGAAAGGATTGTTGAGTAGTATTGAATGAAGGT 3' +
B: 5'          TCTACACTTCTTATCCCCCAACCTTCATTCAATACTACTCAACA 3' +
C: 5'          AAAACACTCATAAAATTTTCTCATCCTTTCTCTTCCACTTAACT 3'

Step 2. Reverse (except Strand A)
A: 5' GATGAGAAGTTAAGTGGAAGAGAAAGGATTGTTGAGTAGTATTGAATGAAGGT 3' +
B: 3'          ACAACTCATCATAACTTACTTCCAACCCCCTATTCTTCACATCT 5' +
C: 3'          TCAATTCACCTTCTCTTTCCTACTCTTTTAAAATACTCACAAAA 5'

Step 3. Fit

find the complement of the 3' -> 5' strand, and search the 5' -> 3' strand for it

start by searching strand A for all of the 3' complement and take one letter away from the end of the search at a time until a match is found (match 1)
then search strand A again but this time remove a letter from the beginning of the search sequence until a match is found (match 2)

whichever match (match 1 or match 2) has the longest match is the best alignment. if it's the same, it will have the same alignment

then align the strands with the rest handing off

5' GATGAGAAGTTAAGTGGAAGAGAAAGGAT-TGTTGAGTAGTATTGAATGAAGGT 3'
3'        TCAATTCACCTTCTCTTTCCTA ACAACTCATCATAACTTACTTCCAACCCCCTATTCTTCACATCT
			       CTCTTTTAAAATACTCACAAAA

Notice how Strand C hangs off because strand B will match at it's designated spot.
In strand A, we use a '-' to represent that the sequences are part of the same strand,
but we use a space to represent the beginning/ending of different strands such as strands B and C in this example alignment

----------------------------------------------
Example 1:
MATCH 1 EXAMPLE - PERFECT FIT (Strand C)
Note: Notice how strand C is able to fit perfectly with strand A and there is no hangoff

structure r0-Gate = AAGTGAAGATTTTATTAATAAAAAGTTGAAAGAGTGATGGTTGTGAATTGGATGAGAAG+CTCTTCCACTTAACTTCTCATCCAATTCACAACCATC+ACTCTTTCAACTTTTTATTAATAAAATC

Step 1. Separate
AAGTGAAGATTTTATTAATAAAAAGTTGAAAGAGTGATGGTTGTGAATTGGATGAGAAG +
                      CTCTTCCACTTAACTTCTCATCCAATTCACAACCATC +
                               ACTCTTTCAACTTTTTATTAATAAAATC

Step 2. Reverse
AAGTGAAGATTTTATTAATAAAAAGTTGAAAGAGTGATGGTTGTGAATTGGATGAGAAG +
                      CTACCAACACTTAACCTACTCTTCAATTCACCTTCTC +
                               CTAAAATAATTATTTTTCAACTTTCTCA

Step 3. Fit
             5' AAGTGAAGATTTTATTAATAAAAAGTTGAAAGAGT-GATGGTTGTGAATTGGATGAGAAG              3'
             3'        CTAAAATAATTATTTTTCAACTTTCTCA CTACCAACACTTAACCTACTCTTCAATTCACCTTCTC 5'


----------------------------------------------
Example 2:
MATCH 2 EXAMPLE

structure r0-Trans_waste = GATGAGAAGTTAAGTGGAAGAGAAAGGATTGTTGAGTAGTATTGAATGAAGGT+CTTCATTCAATACTACTCAACAATCCTTT+CTCTTCCACTTAACTTCTCATCCAATTCACAACCATC

Step 1. Separate
A: 5' GATGAGAAGTTAAGTGGAAGAGAAAGGATTGTTGAGTAGTATTGAATGAAGGT 3' +
B: 5'                         CTTCATTCAATACTACTCAACAATCCTTT 3' +
C: 5'                 CTCTTCCACTTAACTTCTCATCCAATTCACAACCATC 3'

Step 2. Reverse
A: 5' GATGAGAAGTTAAGTGGAAGAGAAAGGATTGTTGAGTAGTATTGAATGAAGGT 3' +
B: 3'                         TTTCCTAACAACTCATCATAACTTACTTC 5' +
C: 3'                 CTACCAACACTTAACCTACTCTTCAATTCACCTTCTC 5'

Step 3. Fit

             5'                GATGAGAAGTTAAGTGGAAGAG-AAAGGATTGTTGAGTAGTATTGAATGAAGGT 3'
             3' CTACCAACACTTAACCTACTCTTCAATTCACCTTCTC TTTCCTAACAACTCATCATAACTTACTTC   5'


"""

from Bio.Seq import Seq
from gui.methods.ConvertUtil import *
import pprint


def separate_strands(structure_sequence):
    """
    Separate the strands of a structure using the '+' symbol.
    :param structure_sequence: Full sequence of the structure.
    :return: List of individual strands.
    """
    return structure_sequence.split("+")


def reverse_complement_strand(strand):
    """
    Reverse complement a DNA strand.
    :param strand: DNA strand sequence.
    :return: Reverse complement of the strand.
    """
    return str(Seq(strand).reverse_complement())


def complement_strand(strand):
    """
    Complement a DNA strand.
    :param strand: DNA strand sequence.
    :return: Complement of the strand.
    """
    return str(Seq(strand).complement())


def calculate_match(sequence1, sequence2, trim_from_end=True):
    """
    Calculate the best match length by trimming sequence2 from either end or start.
    :param sequence1: First sequence (leading strand).
    :param sequence2: Second sequence (lagging strand complement).
    :param trim_from_end: If True, trim sequence2 from the end; otherwise, trim from the start.
    :return: Tuple (offset, match_length).
    """
    max_match_length = 0
    best_offset = 0

    for trim in range(len(sequence2)):
        trimmed_sequence = sequence2[:-trim] if trim_from_end else sequence2[trim:]
        offset = -trim if trim_from_end else trim

        match_length = 0
        for i in range(min(len(sequence1), len(trimmed_sequence))):
            if sequence1[i] == trimmed_sequence[i]:
                match_length += 1
            else:
                break

        if match_length > max_match_length:
            max_match_length = match_length
            best_offset = offset

    return best_offset, max_match_length


def best_match(leading, comp):
    # Match tuples are (leading offset, length of match)

    # match1 removes a character at a time from the end of the sequence
    # match2 removes a character at a time from the beginning of the sequence

    # if the offset is negative, it means we move strand A back a certain amount of positions
    # if the offset is positive, it means we move strand A forward a certain amount of positions

    # whichever match has the highest length is the one we will use

    # Example:
    # Strand A: CTATGTGGGTAATCACC
    # Strand B: TTAGTTTTTTTTTTTT
    # Strand B Complement: AATCAAAAAAAAAAAA

    # match1 would be (-11, 5)
    # match2 would be (14, 2)

    match1 = calculate_match(leading, comp, trim_from_end=True)
    match2 = calculate_match(leading, comp, trim_from_end=False)

    pprint.pp(match1)
    pprint.pp(match2)

    return match1 if match1[1] >= match2[1] else match2


def find_best_alignment(leading, lagging):
    """
    Returns the proper alignment/offsets for leading and lagging strands as well as their connections
    :param leading: the leading DNA strand sequence
    :param leading: the lagging DNA strand sequence
    :return: tuple of leading strand, connection alignment, and lagging strand
    """
    matches = {}
    for strand in lagging:
        comp = complement_strand(strand)

        # Match tuples are (leading offset, length of match)
        matches[strand] = best_match(leading, comp)

    # if the offset is negative, it means we move strand A back a certain amount of positions
    # if the offset is positive, it means we move strand A forward a certain amount of positions

    lead = ""
    align = ""
    lag = ""

    return lead.strip(), align.strip(), lag.strip()


def analyze_structure(structure):
    """
    Analyze the strands within a structure for complementarity.
    :param structure: Piperine Structure object containing strands.
    :return: Dictionary of strand alignments.
    """

    # Step 1: Separate
    strands = separate_strands(structure.Structure)
    pprint.pp(strands)
    if len(strands) < 2:
        # No need to analyze a single-strand structure
        return None

    alignments = {}
    leading_strand = None
    lagging_strand_rev = []

    maxLen = 0
    for s in strands:
        if len(s) > maxLen:
            maxLen = len(s)
            leading_strand = s

    for s in strands:
        if s != leading_strand:
            lagging_strand_rev.append(s)

    pprint.pp(leading_strand)
    pprint.pp(lagging_strand_rev)

    # Note that leading_strand will reamin in the 5' -> 3' orientation
    # We now need to alter lagging_strand_rev's array so they're all 3' -> 5'

    # Step 2: Reverse non-leading strands
    lagging_strand = []

    for s in lagging_strand_rev:
        lagging_strand.append(s[::-1])

    # Step 3: Align
    leading, alignment, lagging = find_best_alignment(leading_strand, lagging_strand)
    alignments["Leading"] = leading
    alignments["Alignment"] = alignment
    alignments["Lagging"] = lagging

    return alignments


def analyze_design_structures(design):
    """
    Analyze all structures in a design.
    :param design: Piperine Design object containing structures.
    :return: Dictionary of analysis results for each structure.
    """
    structure_results = {}

    for structure in design.Structures:
        print(f"Analyzing structure: {structure.Name}")
        results = analyze_structure(structure)
        if results:
            structure_results[structure.Name] = results

    return structure_results


def visualize_alignment(name, alignments):
    """
    Visualize alignments for a structure.
    :param name: Name of the structure.
    :param alignments: Alignment details for strands.
    """
    pprint.pp(alignments)

    print(f"\nAlignments for structure {name}:")
    for alignment in alignments:
        pprint.pp(alignment)
        print(f'{alignment["Leading"]}')
        print(f'{alignment["Alignment"]}')
        print(f'{alignment["Lagging"]}')


def main():
    # Load PiperineOutput
    piperine_output = process_piperine_output('../../../static/reference/piperine/Example 0', logging=False)
    winning = piperine_output.BestDesigns

    for design in piperine_output.Designs:
        if design.Name not in [d.Name for d in winning]:
            continue
        print(f"Analyzing design: {design.Name}")

        # Analyze structures
        all_structure_results = analyze_design_structures(design)

        # Display results
        for structure_name, alignments in all_structure_results.items():
            visualize_alignment(structure_name, alignments)


if __name__ == "__main__":
    main()
