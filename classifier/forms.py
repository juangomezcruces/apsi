from django import forms

class TextClassificationForm(forms.Form):
    text = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 6,
            'placeholder': 'Enter political text to analyze...',
            'maxlength': 2000
        }),
        max_length=2000,
        help_text='Enter up to 2000 characters of political text for analysis.'
    )