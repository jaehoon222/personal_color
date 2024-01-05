from django.db import models

# Create your models here.
class Picture(models.Model):
    data = models.ImageField()

class PersonalColor(models.Model):
    color = models.CharField(max_length=100)

class Comment(models.Model):
    color = models.ForeignKey(PersonalColor,on_delete=models.CASCADE)
    contents = models.TextField()
    created_date = models.DateField(auto_created=True)

