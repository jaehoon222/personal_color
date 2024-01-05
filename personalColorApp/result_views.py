from django.shortcuts import render, get_object_or_404

from personalColorApp.models import PersonalColor


def color_1(request):

    color = get_object_or_404(PersonalColor, color="color_1")

    context = {'color':color}
    return render(request,"personalColor/color_1.html",context)


def color_2(request):
    color = get_object_or_404(PersonalColor, color="color_2")
    context = {'color':color}
    return render(request,"personalColor/color_2.html",context)

def color_3(request):
    context = {'color':"color_3"}
    return render(request,"personalColor/color_3.html",context)

