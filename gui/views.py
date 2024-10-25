from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_protect

from gui.methods.FSMUtil import *
from gui.methods.ConvertUtil import *
from gui.methods.FUNDNAUtil import *
from gui.classes import *
from gui.classes.fsm import *

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

        # SECTION 4: DSD
        # Handle CRN/DSD conversion using Nuskell
        try:
            # Create CRN object
            if crn is not None:
                # Run Nuskell and get the temp directory
                temp_dir = run_piperine(crn)

                print(f"piperine temp directory: {temp_dir}")

                # Process the piperine output files
                piperine_output = process_piperine_output(temp_dir)

                print(piperine_output)

                # Cleanup the temporary directory
                cleanup_temp_dir(temp_dir)
        except Exception as err:
            return HttpResponse(f"Error processing Piperine: {str(err)}")

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
