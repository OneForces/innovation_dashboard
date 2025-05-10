from django.db import models

class UploadedFile(models.Model):
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class RegionData(models.Model):
    region = models.CharField(max_length=255)
    year = models.IntegerField()
    indicator_name = models.CharField(max_length=255)
    value = models.FloatField()
