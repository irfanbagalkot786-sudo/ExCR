from django.db import models
from django.contrib.auth.models import User

class VideoUpload(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    title = models.CharField(max_length=200, blank=True)
    video_file = models.FileField(upload_to='videos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    processed_at = models.DateTimeField(null=True, blank=True)
    engagement_score = models.FloatField(null=True, blank=True)
    student_count = models.IntegerField(default=0)
    attentive_pct = models.FloatField(default=0.0)
    sleepy_pct = models.FloatField(default=0.0)
    distracted_pct = models.FloatField(default=0.0)
    neutral_pct = models.FloatField(default=0.0)
    heatmap_image = models.ImageField(upload_to='heatmaps/', null=True, blank=True)
    
    def __str__(self):
        return f"{self.title} ({self.uploaded_at})"

class WebcamSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    engagement_score = models.FloatField()
    attentive = models.IntegerField()
    sleepy = models.IntegerField()
    distracted = models.IntegerField()
    neutral = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Webcam Session {self.id} ({self.created_at})"
