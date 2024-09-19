from django.shortcuts import render
from django.http import HttpResponse

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
"""


# Create your views here.
def index(request):
    return render(request, 'index.html')


def guiIndex(request):
    return render(request, 'gui/index.html')


def convert(request):
    return render(request, 'gui/convert.html')


def convertResult(request):
    # Extract GET parameters
    from_level = request.GET.get('from_level', 'default_value')  # Provide a default value if needed
    to_level = request.GET.get('to_level', 'default_value')

    # Pass the variables to the template
    context = {
        'from_level': from_level,
        'to_level': to_level,
    }

    return render(request, 'gui/convertResult.html', context)


def simulate(request):
    return render(request, 'gui/simulate.html')


def simulateResult(request):
    return render(request, 'gui/simulateResult.html')


def latexEditor(request):
    return render(request, 'templates/partials/latex_editor.html')


# def index(request):
#    return HttpResponse("Hello, world. You're at the polls index.")
