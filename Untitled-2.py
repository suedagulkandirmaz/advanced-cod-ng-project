import tensorflow as tf
import PIL import Image

image_size = (224, 224)
batch_size = 32

class Plant:
    def_init_(self, name):
    self.name = name

    def info(self):
        retur f"{self.name} is not a disease."

class DiseasePlant(Plant):
    def_init_(self,name,disease):
    super()._init_(name)
    self.disease = DiseasePlant

    def info(self):
        return f"{self.name} has disease: {self.disease}."




from models.plant import Plant 

class User(Person):
    def__init__(self, person_id, name):
       super().__init__(person_id, name):
       self.ansalyzed_plants = []
    
    def add_plant_analysis(self, plant: Plant):
        self.analyzed_plants.append(plant)

    