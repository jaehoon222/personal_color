from django import forms
from .models import Picture, Comment


class PictureForms(forms.ModelForm):
    class Meta:
        model = Picture
        fields = ['data']

class CommentForms(forms.ModelForm):
    class Meta:
        model = Comment
        fields = ['contents']