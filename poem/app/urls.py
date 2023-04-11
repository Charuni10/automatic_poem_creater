from . import views
from django.urls import path

urlpatterns = [
    path('', views.home, name='home'),
    path('pred/', views.pred, name='pred'),

]