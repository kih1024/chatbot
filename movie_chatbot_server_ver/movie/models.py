from django.db import models

class User_state(models.Model):
	user_key=models.CharField(max_length=30,default="empty")
	user_state=models.IntegerField(default=0)
	sameName=models.IntegerField(default=0)
	nameTem=models.CharField(max_length=30,default="empty")

	def __str__(self):
		return u'%s %s' %(self.user_key, self.nameTem)

class Picture(models.Model):
       pic=models.ImageField(upload_to = 'media/')


# Create your models here.
