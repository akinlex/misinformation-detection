from django.urls import  path
from . import views 

app_name = 'vericovid'

urlpatterns = [
    path('', views.index, name='index')
]