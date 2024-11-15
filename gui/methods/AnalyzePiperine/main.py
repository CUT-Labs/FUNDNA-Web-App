from Bio.Seq import Seq
from gui.methods.ConvertUtil import *
import pprint


def find_matches(seqA, seqB, tolerance=0):
    """
    Find all contiguous matches of Strand B within Strand A.
    :param seqA: Sequence of the longer strand (5' -> 3').
    :param seqB: Sequence of the shorter strand (5' -> 3').
    :param tolerance: Percent match that the short strand must get for a match to be recorded
    :return: Dictionary where keys are starting positions in Strand A,
             and values are tuples (startB, length).
    """
    results = {}

    for i, nucA in enumerate(seqA):  # Iterate over each nucleotide in Strand A
        startB = None
        length = 0

        for j, nucB in enumerate(seqB):  # Iterate over each nucleotide in Strand B
            if i + j < len(seqA) and seqA[i + j] == nucB:
                if startB is None:
                    startB = j  # Record the starting position in Strand B
                length += 1  # Increment the match length
            else:
                # If there's a mismatch or end of strand, finalize the match
                if length > 0:
                    if length >= len(seqB) * tolerance:
                        results[i] = (startB, length)
                break

        # If the loop ends with a match, record it
        if length > 0:
            if length >= len(seqB) * tolerance:
                results[i] = (startB, length)

    return results


def visualize_alignment(strand1_name, strand2_name, strand1, strand2, match):
    """
    Visualize the alignment between two strands.
    :param strand1_name: Name of the first strand.
    :param strand2_name: Name of the second strand.
    :param strand1: Sequence of the first strand.
    :param strand2: Sequence of the second strand (reverse complement).
    :param match: Tuple with start and end indices of the match (start1, length).
    """
    start1, (start2, length) = match

    # Get the matched segments
    segment1 = strand1[start1:start1 + length]
    segment2 = strand2[start2:start2 + length][::-1]  # Reverse complement for alignment visualization

    # Generate alignment visualization
    alignment_str = []
    for b1, b2 in zip(segment1, segment2):
        if b1 == "A" and b2 == "T" or b1 == "T" and b2 == "A" or b1 == "C" and b2 == "G" or b1 == "G" and b2 == "C":
            alignment_str.append("|")
        else:
            alignment_str.append(" ")

    # Format and print the results
    print(f"  Alignment between {strand1_name} and {strand2_name}:")
    print(f"  Strand 1: {strand1}")
    print(f"            {' ' * start1}{''.join(alignment_str)}")
    print(f"  Strand 2: {' ' * start1}{strand2[::-1]}\n")  # Reverse complement visualization


def analyze_complex_strands(complex):
    """
    Analyze the strands within a complex to find complementarity.
    :param complex: Piperine Complex object containing strands.
    :return: Dictionary of strand pair matches and alignments.
    """
    results = {}
    strands = {strand.Name: strand.Strand for strand in complex.Strands}

    strand_names = list(strands.keys())

    for i in range(len(strand_names)):
        for j in range(i + 1, len(strand_names)):
            name1 = strand_names[i]
            name2 = strand_names[j]
            strand1 = strands[name1]
            strand2 = strands[name2]

            # Find matches
            matches = find_matches(strand1, strand2, tolerance=0)
            if matches:
                results[(name1, name2)] = matches

    return results


def analyze_design_complexes(design):
    """
    Analyze all complexes in a design.
    :param design: Piperine Design object containing complexes.
    :return: Dictionary of analysis results for each complex.
    """
    complex_results = {}

    for complex in design.Complexes:
        if complex == design.Complexes[0]:
            print(f"Analyzing complex: {complex.Name}")
            results = analyze_complex_strands(complex)
            if results:
                complex_results[complex.Name] = results

    return complex_results


def main():
    # Load PiperineOutput
    piperine_output = process_piperine_output('../../../static/reference/piperine/Example 0', logging=False)
    winning = piperine_output.BestDesigns

    for design in piperine_output.Designs:
        if design.Name not in [d.Name for d in winning]:
            continue
        print(f"Analyzing design: {design.Name}")

        # Analyze complexes
        all_complex_results = analyze_design_complexes(design)

        # Display results
        for complex_name, results in all_complex_results.items():
            print(f"\nComplementarity results for complex {complex_name}:")

            strands = {}
            for complex in design.Complexes:
                if complex.Name == complex_name:
                    for strand in complex.Strands:
                        strands[strand.Name] = strand.Strand

            pprint.pp(results)

            for (strand1_name, strand2_name), matches in results.items():
                for match in matches.items():
                    visualize_alignment(
                        strand1_name, strand2_name,
                        strands[strand1_name], strands[strand2_name], match
                    )


if __name__ == "__main__":
    main()
