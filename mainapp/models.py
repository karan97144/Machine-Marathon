from django.db import models
from django.db.models.fields.files import FileField

# Create your models here.
class Fileupload(models.Model):
    file = models.FileField(upload_to = "media")