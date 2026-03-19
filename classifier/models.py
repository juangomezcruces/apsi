# classifier/models.py
from django.db import models

class ClassificationResult(models.Model):
    """Stores a classification result when the user opts in via the save toggle."""
    input_text = models.TextField()
    scores = models.JSONField(default=dict)
    selected_approaches = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
 
    class Meta:
        ordering = ['-created_at']
 
    def __str__(self):
        return f"Result {self.id} – {self.created_at.strftime('%Y-%m-%d %H:%M')}"