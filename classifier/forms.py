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
    
    # Left-Right approaches
    left_right_direct = forms.BooleanField(
        required=False,
        initial=True,
        label='Left-Right (Direct Regression)',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    left_right_hypothesis = forms.BooleanField(
        required=False,
        initial=True,
        label='Left-Right (Hypothesis-Based)',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    # Liberal-Illiberal approaches
    liberal_illiberal_direct = forms.BooleanField(
        required=False,
        initial=True,
        label='Liberal-Illiberal (Direct Regression)',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    liberal_illiberal_hypothesis = forms.BooleanField(
        required=False,
        initial=True,
        label='Liberal-Illiberal (Hypothesis-Based)',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    # Populism approaches
    populism_direct = forms.BooleanField(
        required=False,
        initial=True,
        label='Populism (Direct Regression)',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    populism_hypothesis = forms.BooleanField(
        required=False,
        initial=True,
        label='Populism (Hypothesis-Based)',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
        # NEW: Left-Right Response-Based
    left_right_responses = forms.BooleanField(
        required=False,
        initial=True,
        label='Left-Right (Response-Based)',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    # NEW: Liberal-Illiberal Response-Based
    liberal_illiberal_responses = forms.BooleanField(
        required=False,
        initial=True,
        label='Liberal-Illiberal (Response-Based)',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    # NEW: Populism-Pluralism Response-Based
    populism_responses = forms.BooleanField(
        required=False,
        initial=True,
        label='Populism-Pluralism (Response-Based)',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

# Also update your clean() method to include these new fields:
def clean(self):
    cleaned_data = super().clean()
    
    # Check that at least one approach is selected - ADD the new fields
    selected_approaches = [
        # Direct approaches
        cleaned_data.get('left_right_direct'),
        cleaned_data.get('liberal_illiberal_direct'), 
        cleaned_data.get('populism_direct'),
        
        # Hypothesis-based approaches  
        cleaned_data.get('left_right_hypothesis'),
        cleaned_data.get('liberal_illiberal_hypothesis'),
        cleaned_data.get('populism_hypothesis'),
        
        # NEW: Response-based approaches
        cleaned_data.get('left_right_responses'),
        cleaned_data.get('liberal_illiberal_responses'),
        cleaned_data.get('populism_responses'),
    ]
    
    if not any(selected_approaches):
        raise forms.ValidationError("Please select at least one analysis approach.")
    
    return cleaned_data