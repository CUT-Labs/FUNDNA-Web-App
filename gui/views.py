from django.shortcuts import render
from django.http import HttpResponse


# Create your views here.
def index(request):
    return render(request, 'index.html')  # This will extend from base.html

# def index(request):
#    return HttpResponse("Hello, world. You're at the polls index.")
