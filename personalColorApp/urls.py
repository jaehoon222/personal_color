from django.urls import path

from personalColorApp import views, result_views

app_name = 'personalColor'
urlpatterns = [
    path('', views.index,name="index"),
    path('picture/',views.picture,name="picture"),
    path('result/',views.result_view,name="result"),
    # path('result/color_2/',result_views.color_2,name="color_2"),
    # path('result/color_3/',result_views.color_3,name="color_3"),
    # path('comment/create/<str:color>',views.create_comment,name="create_comment")
    # # path('/picture/get',views.getPicture,name="getPicture"),
]