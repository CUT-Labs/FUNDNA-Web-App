from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_protect

from gui.methods.FSMUtil import *
from gui.methods.ConvertUtil import *
from gui.methods.FUNDNAUtil import *
from gui.classes import *
from gui.classes.fsm import *

import re

import gui as gui

import sympy
import sympy as sp
from sympy import sympify

"""
    EXAMPLE:
        def viewName(request):
            return render(request, 'path/from/pages')
            
    path/from/pages could be anything inside of pages. for example,
    
    templates/base.html
    index.html
    gui/index.html
    gui/convert.html
    
    etc.
    
    the viewName is what is called in urls.py and where the http path is defined.
    path/from/pages does nothing except find the file and use that as the html source for render()    
    
    redirect('name') # name is pulled from urls.py name=
"""


# Create your views here.
def index(request):
    return render(request, 'index.html')


def guiIndex(request):
    return render(request, 'gui/index.html')


def convert(request):
    return render(request, 'gui/convert.html')


@csrf_protect
def convertResult(request):
    if request.method == 'POST':
        from_level = request.POST.get('FromLevel')
        to_level = request.POST.get('ToLevel')
        latex_input = request.POST.get('LaTeX_Input', '')
        crn_dsd_input = request.POST.get('CRN_DSD_Input', '')

        # Define sections based on from and to levels
        sections = {
            ("function", "gate"): ["function", "gate"],
            ("function", "crn"): ["function", "gate", "crn"],
            ("function", "dsd"): ["function", "gate", "crn", "dsd"],
            ("function", "dna"): ["function", "gate", "crn", "dsd", "dna"],
            ("gate", "crn"): ["gate", "crn"],
            ("gate", "dsd"): ["gate", "crn", "dsd"],
            ("gate", "dna"): ["gate", "crn", "dsd", "dna"],
            ("crn", "dsd"): ["crn", "dsd"],
            ("crn", "dna"): ["crn", "dsd", "dna"],
            ("dsd", "dna"): ["dsd", "dna"],
        }

        pointEstimation = float(request.POST.get('PointEstimation', 0))
        degreeEstimation = int(request.POST.get('DegreeEstimation', 5))

        scheme = request.POST.get('nuskell_scheme', 'soloveichik2010.ts')
        verify = request.POST.get('nuskell_verify', False)

        selected_sections = sections.get((from_level, to_level), [])

        function = None
        graph_url = None

        print(selected_sections)
        print('--')
        print(latex_input)

        taylorStr = ""
        rearrangement = ""

        gate_url = ""
        gate_information = ""

        crn = None
        crn_table = ""

        rearrangementType = None

        estimation = 0

        nuskell_output = None
        piperine_output = None

        # SECTION 1: FUNCTION
        # Approximate functions with taylor series to the degree
        # and return the point trace and est. function
        try:
            if 'function' in selected_sections and latex_input:
                print("Loading function approximation...")

                function_lambda, graph_url = functionData(latex_input)

                print("\n\nStep 1:\tGenerate Function Object and Rearrangements\n\n")
                function = gui.classes.Function(latex_input,
                                                function_lambda,
                                                pointEstimation,
                                                degreeEstimation + 1,
                                                determineFunctionType(latex_input),
                                                f'FUNDNA Approximation - Point: {pointEstimation} - Degree: {degreeEstimation}',
                                                "x")

                function.generateCoeffs()

                print(f'Taylor Approximation: {function.taylorString}')
                taylorStr = sympy.latex(sympify(function.taylorString))
                print(f'Taylor Approximation (LaTeX): {taylorStr}')

                print(f'Rearrangement: {function.rearrangeString}')
                rearrangement = sympy.latex(sympify(function.rearrangeString))
                print(f'Rearrangement (LaTeX): {rearrangement}')

                rearrangementType = function.rearrangeType.value
                estimation = function.function(pointEstimation)

                if rearrangement is not None:
                    lExpress = "lambda " + function.variable + ": " + function.rearrangeString
                    # lambda x: sin(x)
                    rearrLambda = eval(lExpress)

                    graph_url = graphOriginalAndRearrangement(LatexToLambda(latex_input),
                                                              rearrLambda,
                                                              pointEstimation,
                                                              degreeEstimation)

        except Exception as err:
            return HttpResponse(f"Error processing function: {str(err)}")

        # SECTION 2: GATE
        try:
            if function is not None and function.rearrangeType is not RearrangeType.UNKNOWN:
                print(f'\n\nStep 2:\tGenerating Gates...\n\n')
                gate_url, gate_information = function.generateCircuit()

                for gate in gate_information:
                    PrintGateInfo(gate)

                print(gate_url)
            else:
                gate_information = ["Feature not yet implemented!"]
        except Exception as err:
            return HttpResponse(f"Error processing gates: {str(err)}")

        # SECTION 3: CRN
        try:
            if function is not None:
                crn, crn_table = function.generateReactions()
            elif crn_dsd_input is not None and from_level == 'crn':
                crn_table = crn_dsd_input
                crn = CRN()

                for reaction in crn_table.split(";"):
                    if reaction != "":
                        for r in reaction.split("\n"):
                            print(r)
                            crn.AddReaction(r)
        except Exception as err:
            return HttpResponse(f"Error processing CRN: {str(err)}")

        # SECTION 4: DSD
        # Handle CRN/DSD conversion using Nuskell
        if "dsd" in selected_sections:
            try:
                # Create CRN object
                if crn is not None:
                    # Run Nuskell and get the temp directory
                    temp_dir = run_nuskell(crn, scheme, verify)

                    # Process the Nuskell output files
                    nuskell_output = process_nuskell_output(temp_dir)

                    print(nuskell_output)

                    # Cleanup the temporary directory
                    cleanup_temp_dir(temp_dir)
            except Exception as err:
                return HttpResponse(f"Error processing Nuskell: {str(err)}")

        # SECTION 5: DNA
        # Handle CRN/DNA conversion using Piperine
        transposed_scores = []
        debug = True
        if "dna" in selected_sections:
            if debug:
                # Initialize PiperineOutput
                piperine_output = PiperineOutput()

                # Define raw scores, rank array, fractional excess, and percent badness for each design
                raw_scores_data = [
                    [0.05, 0.07, 4.23, 11.88, 51.00, 7.00, 0.40, 0.96, 0.43, 0.95, 5.50, 3.31, 1.00, 230.00, 0.00,
                     2761.00, 358.50, 360.14, 0.02, 0.39],
                    [0.05, 0.09, 5.34, 13.25, 49.00, 6.00, 0.40, 0.96, 0.40, 0.94, 5.33, 3.16, 6.00, 270.00, 0.00,
                     2898.00, 386.50, 399.23, 0.02, 0.39],
                    [0.04, 0.06, 9.33, 20.35, 29.00, 6.00, 0.42, 0.97, 0.42, 0.96, 7.02, 3.50, 4.00, 218.00, 0.00,
                     2999.00, 356.50, 375.44, 0.02, 0.39]
                ]

                rank_array_data = [
                    [1, 1, 0, 0, 2, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0],
                    [2, 2, 1, 1, 1, 0, 2, 2, 2, 2, 0, 0, 2, 2, 0, 1, 2, 2, 0, 0],
                    [0, 0, 2, 2, 0, 0, 0, 0, 1, 0, 2, 2, 1, 0, 0, 2, 0, 1, 0, 0]
                ]

                fractional_excess_data = [
                    [0.09, 0.08, 0.00, 0.00, 0.76, 0.17, 0.04, 0.01, 0.00, 0.00, 0.03, 0.05, 0.00, 0.06, 0.00, 0.00,
                     0.01, 0.00, 0.14, 0.00],
                    [0.23, 0.41, 0.26, 0.12, 0.69, 0.00, 0.05, 0.01, 0.08, 0.02, 0.00, 0.00, 5.00, 0.24, 0.00, 0.05,
                     0.08, 0.11, 0.00, 0.00],
                    [0.00, 0.00, 1.20, 0.71, 0.00, 0.00, 0.00, 0.00, 0.03, 0.00, 0.32, 0.11, 3.00, 0.00, 0.00, 0.09,
                     0.00, 0.04, 0.00, 0.00]
                ]

                percent_badness_data = [
                    [40.99, 20.34, 0.00, 0.00, 100.00, 100.00, 79.91, 77.04, 0.00, 27.22, 10.01, 43.72, 0.00, 23.08,
                     0.00, 0.00, 6.67, 0.00, 100.00, 0.00],
                    [100.00, 100.00, 21.65, 16.15, 90.91, 0.00, 100.00, 100.00, 100.00, 100.00, 0.00, 0.00, 100.00,
                     100.00, 0.00, 57.56, 100.00, 100.00, 0.00, 0.00],
                    [0.00, 0.00, 100.00, 100.00, 0.00, 0.00, 0.00, 0.00, 37.50, 0.00, 100.00, 100.00, 60.00, 0.00, 0.00,
                     100.00, 0.00, 39.16, 0.00, 0.00]
                ]

                # Populate PiperineOutput with designs and score arrays
                for i in range(3):
                    design = Design(f"Design {i + 1}")

                    # Assign RawScores, RankArray, FractionalExcessArray, and PercentBadnessArray for each design
                    design.RawScores.from_list(raw_scores_data[i])
                    design.RankArray.from_list(rank_array_data[i])
                    design.FractionalExcessArray.from_list(fractional_excess_data[i])
                    design.PercentBadnessArray.from_list(percent_badness_data[i])

                    # Add Sequences, Strands, Structures, SignalStrands, and Complexes with sample data
                    design.Sequences = [
                        Sequence("Seq1", "ATGCAT"),
                        Sequence("Seq2", "TACGCG")
                    ]

                    design.Strands = [
                        Strand("Strand1", "ATGCGT", False),
                        Strand("Strand2", "TACGGA", False)
                    ]

                    design.Structures = [
                        Structure("Struct1", "TACGGA"),
                        Structure("Struct2", "ATGCGT"),
                        Structure("Struct3", "ATGCGT+TACGGATACGGA+ATGCGTATGCGT"),
                        Structure("Struct4", "A...G...T")
                    ]

                    design.SignalStrands = [
                        Strand("Signal1", "ATGCC", True),
                        Strand("Signal2", "CGTAA", True)
                    ]

                    design.Complexes = [
                        Complex("Complex1", [Strand("Strand1", "ATGCGT", False), Strand("Strand2", "TACGGA", False)],
                                True)
                    ]

                    # Append each design to PiperineOutput
                    piperine_output.Designs.append(design)
                piperine_output.MetaRanksArray()
            else:
                try:
                    # Create CRN object
                    if crn is not None:
                        # Run piperine and get the temp directory
                        temp_dir = run_piperine(crn)

                        print(f"Piperine Temp Directory: {temp_dir}")

                        # Process the piperine output files
                        piperine_output = process_piperine_output(temp_dir)

                        print(piperine_output)

                        # Cleanup the temporary directory
                        cleanup_temp_dir(temp_dir)
                except Exception as err:
                    return HttpResponse(f"Error processing Piperine: {str(err)}")

            if piperine_output:
                _, ranked_meta_scores, _, _ = piperine_output.MetaRanksArray()

                # Transpose ranked_meta_scores for template use
                design_names = ranked_meta_scores["Design"]
                num_designs = len(design_names)

                for i in range(num_designs):
                    design_row = [
                        design_names[i],  # Design name
                        ranked_meta_scores["Meta Sum"][i],
                        ranked_meta_scores["Worst-rank"][i],
                        ranked_meta_scores["Worst-weighted-rank"][i],
                        ranked_meta_scores["Sum-of-ranks"][i],
                        ranked_meta_scores["Weighted Sum-of-ranks"][i],
                        ranked_meta_scores["Fractional-excess"][i],
                        ranked_meta_scores["Weighted Fractional-excess"][i],
                        ranked_meta_scores["Percent-badness"][i],
                        ranked_meta_scores["Weighted Percent-badness"][i],
                    ]
                    transposed_scores.append(design_row)

        context = {
            'from_level': from_level,
            'to_level': to_level,
            'selected_sections': selected_sections,

            'latex_input': latex_input,
            'graph_url': graph_url,

            'maclaurin_approximation': taylorStr,
            'rearrangement_type': rearrangementType,
            'rearrangement': rearrangement,

            'point': pointEstimation,
            'estimation': estimation,

            'gate_url': gate_url,
            'gate_information': gate_information,

            'crn': crn,
            'crn_table': crn_table,

            'nuskell_output': nuskell_output,  # Includes enum_data, sys_data, and log_data
            'piperine_output': piperine_output,
            'transposed_scores': transposed_scores,
        }

        return render(request, 'gui/convertResult.html', context)

    return redirect('convert')


def simulate(request):
    return render(request, 'gui/simulate.html')


@csrf_protect
def simulateResult(request):
    if request.method == 'POST':
        latex_function = request.POST.get('LaTeX_Input')
        apply_bernstein = request.POST.get('applyBernstein') == 'on'
        bernstein_degree = int(request.POST.get('bernsteinDegree')) if apply_bernstein else None
        function_name = request.POST.get('functionName', 'Function')

        print(latex_function)
        print(apply_bernstein)
        print(bernstein_degree)
        print(function_name)
        # Convert LaTeX function to Python lambda
        function = LatexToLambda(latex_function)

        # Create FSM and perform simulation
        fsm = FSMSolver(function, function_name, equation=latex_function, variable="x", logging=True, save=False,
                        roundUp=False)

        fsms = []

        fsms.append({
            'name': fsm.name,
            'latex_equation': fsm.original,
            'states': fsm.states.items(),
            'graph': FSMPlot(solverToFSM(fsm), function),
            'latex_objective_function': generate_objective_function_latex(fsm.h_matrix,
                                                                          fsm.b_vector,
                                                                          fsm.c_vector)
        })

        bernstein_approx = None
        fsm_bernstein = None
        # Apply Bernstein approximation if requested
        if apply_bernstein and bernstein_degree:
            bernstein_approx = Bernstein(function, degree=bernstein_degree).construct_bernstein_polynomial()
            fsm_bernstein = FSMSolver(bernstein_approx, f'{function_name} - Bernstein ({bernstein_degree} deg.)',
                                      equation=latex_function, variable="x", logging=True, save=False, roundUp=False)

            fsms.append({
                'name': fsm_bernstein.name,
                'latex_equation': fsm_bernstein.original,
                'states': fsm_bernstein.states.items(),
                'graph': FSMPlot(solverToFSM(fsm_bernstein), bernstein_approx),
                'latex_objective_function': generate_objective_function_latex(fsm_bernstein.h_matrix,
                                                                              fsm_bernstein.b_vector,
                                                                              fsm_bernstein.c_vector)
            })

        context = {
            'function_name': function_name,
            'fsms': fsms
        }
        return render(request, 'gui/simulateResult.html', context)

    return redirect('simulate')


def latexEditor(request):
    return render(request, 'templates/partials/latex_editor.html')


def fsm_result(request):
    return render(request, 'templates/partials/fsm_result.html')

# def index(request):
#    return HttpResponse("Hello, world. You're at the polls index.")
