from django import forms
from .models import PlantImage



class UploadImageForm(forms.Form):
    image = forms.ImageField()

