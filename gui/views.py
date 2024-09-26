from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_protect

from gui.methods.ConvertUtil import *
from gui.methods.FUNDNAUtil import *
from gui.classes import *

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

        pointEstimation = request.POST.get('PointEstimation', 0)
        degreeEstimation = request.POST.get('DegreeEstimation', 5)

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
                                                float(pointEstimation),
                                                int(degreeEstimation) + 1,
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

        except Exception as e:
            return HttpResponse(f"Error processing function: {str(e)}")

        # SECTION 2: GATE
        if function is not None and function.rearrangeType is not RearrangeType.UNKNOWN:
            print(f'\n\nStep 2:\tGenerating Gates...\n\n')
            gate_url, gate_information = function.generateCircuit()

            for gate in gate_information:
                PrintGateInfo(gate)

            print(gate_url)
        else:
            gate_information = ["Feature not yet implemented!"]

        context = {
            'from_level': from_level,
            'to_level': to_level,
            'selected_sections': selected_sections,

            'latex_input': latex_input,
            'graph_url': graph_url,

            'maclaurin_approximation': taylorStr,
            'rearrangement_type': function.rearrangeType.value,
            'rearrangement': rearrangement,

            'point': pointEstimation,
            'estimation': function.function(float(pointEstimation)),

            'gate_url': gate_url,
            'gate_information': gate_information,

            'crn_dsd_input': crn_dsd_input,
        }

        return render(request, 'gui/convertResult.html', context)

    return redirect('convert')


def simulate(request):
    return render(request, 'gui/simulate.html')


def simulateResult(request):
    return render(request, 'gui/simulateResult.html')


def latexEditor(request):
    return render(request, 'templates/partials/latex_editor.html')

# def index(request):
#    return HttpResponse("Hello, world. You're at the polls index.")
