from django.contrib import admin
from django.urls import path
from identify import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('identify/', views.identify_plant, name='identify_plant'),
]