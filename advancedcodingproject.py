import os
import TensorFlow as tf
import numpy as np
import PIL 
import images
import sqlite3
import flask
import ImageDataGenerator


oop
class DiseaseDetector:
    def __init__(self, name):
        self.name = name

    def info(self):
        return f"{self.name} is a healthy."

class DiseasePlant(Plant):
    def __init__(self, name, disease):
        super().__init__(name)
    self.disease = disease

    def info(self):
        return f"{self.name} has disease: {self.disease}."








app = flask(__name__)

classes = ['Tomato', 'Potato', 'Pepper']
statuses = ['Healthy', 'Diseased']
diseased = [Bacterial_spot, ]
for plant in classes:
    for status in statuses:
        path = os.path.join('data', 'dataset', plant, status)
        os.makedirs(path, exist_ok=True)












    