# Generated by Django 2.0.4 on 2018-05-04 07:50

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='User_state',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user_key', models.CharField(default='empty', max_length=30)),
                ('user_state', models.IntegerField(default=0)),
                ('sameName', models.IntegerField(default=0)),
                ('nameTem', models.CharField(default='empty', max_length=30)),
            ],
        ),
    ]