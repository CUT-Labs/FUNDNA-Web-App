from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("gui/", views.guiIndex, name="index"),
    path("gui/convert", views.convert, name="convert"),
    path("gui/simulate", views.simulate, name="simulate"),
    path("gui/convert/result", views.convert, name="convertResult"),
    path("gui/simulate/result", views.simulate, name="simulateResult"),
]