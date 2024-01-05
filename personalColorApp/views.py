from django.utils import timezone

from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404

from .models import PersonalColor,Comment
import cv2
import numpy as np
from .function import eval


# Create your views here.
def index(request):
    return render(request,"personalColor/index.html")



def picture(request):
    # return render(request, "personalColor/Picture.html")

    if request.method == 'POST':
        image_file = request.FILES['image']
        # image_path = settings.MEDIA_ROOT + '/photos/' + image.name
        # with open(image_path, 'wb') as file:
        #     for chunk in image.chunks():
        #         file.write(chunk)
        #=========================================================

        image_data = image_file.read()

        nparr = np.frombuffer(image_data, np.uint8)

        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        val = eval.perscol(image)
        print(val)
        if val is None:
            return render(request, "personalColor/Picture.html")
        # =========================================================
        data = {"color": val}
        return JsonResponse(data)
    else:
        return render(request, "personalColor/Picture.html")

def result_view(request):
    color = request.GET.get('color')  # 쿼리 매개변수 "color" 가져오기
    # 가져온 "color" 값을 사용하여 필요한 작업 수행
    # ...
    print(color)
    return render(request, 'personalColor/color_1.html', {'color': color})

def create_comment(request,color):

    c = get_object_or_404(PersonalColor, color=color)

    if request.method == "POST":
            comment = Comment(color=c,contents=request.POST["content"],created_date=timezone.now())
            comment.save()
            target = "personalColor:"+c.color
            return redirect(target)




