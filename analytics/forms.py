# analytics/forms.py
from django import forms
from .models import UploadedFile

class ExcelUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedFile
        fields = ['file']
