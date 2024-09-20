from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_protect

from gui.methods.ConvertUtil import *

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
            ("function", "crn"): ["function", "crn"],
            ("function", "dsd"): ["function", "crn", "dsd"],
            ("function", "dna"): ["function", "crn", "dsd", "dna_sequence"],
            ("gate", "crn"): ["gate", "crn"],
            ("gate", "dsd"): ["gate", "crn", "dsd"],
            ("gate", "dna"): ["gate", "crn", "dsd", "dna"],
            ("crn", "dsd"): ["crn", "dsd"],
            ("crn", "dna"): ["crn", "dsd", "dna"],
            ("dsd", "dna"): ["dsd", "dna"],
        }

        selected_sections = sections.get((from_level, to_level), [])

        function_data = []
        graph_url = None

        print(selected_sections)
        print('--')
        print(latex_input)

        try:
            if 'function' in selected_sections and latex_input:
                print("Loading function approximation...")

                # Convert LaTeX to lambda and generate points
                function_lambda = LatexToLambda(latex_input)
                x_values, y_values = generatePoints(function_lambda)

                # Generate graph image
                graph = generateGraph(x_values, y_values)

                # Encode graph in base64 for rendering in the template
                graph_url = "data:image/svg+xml;base64," + base64.b64encode(graph.getvalue()).decode()

        except Exception as e:
            return HttpResponse(f"Error processing function: {str(e)}")

        context = {
            'from_level': from_level,
            'to_level': to_level,
            'latex_input': latex_input,
            'function_data': function_data,
            'graph_url': graph_url,  # Pass the graph URL to the template
            'crn_dsd_input': crn_dsd_input,
            'selected_sections': selected_sections
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
