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
        <div class="alert alert-info mt-4" role="alert" style="max-width: 980px; margin: 0 auto;">
          <strong>Important:</strong> "Users are solely responsible for the content they submit, including ensuring compliance with copyright and other applicable legal requirements."
        </div>

    )
    
    # Only hypothesis-based approaches
    left_right_hypothesis = forms.BooleanField(
        required=False,
        initial=True,
        label='Left-Right (Hypothesis-Based)',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    liberal_illiberal_hypothesis = forms.BooleanField(
        required=False,
        initial=True,
        label='Liberal-Illiberal (Hypothesis-Based)',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    populism_hypothesis = forms.BooleanField(
        required=False,
        initial=True,
        label='Populism (Hypothesis-Based)',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    def clean(self):
        cleaned_data = super().clean()
        
        # Check that at least one approach is selected
        selected_approaches = [
            cleaned_data.get('left_right_hypothesis'),
            cleaned_data.get('liberal_illiberal_hypothesis'),
            cleaned_data.get('populism_hypothesis'),
        ]
        
        if not any(selected_approaches):
            raise forms.ValidationError("Please select at least one analysis approach.")
        
        return cleaned_data
