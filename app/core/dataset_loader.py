# data_loader.py
import os
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader

class PlantDiseaseDataLoader:
    def __init__(self, root_path, batch_size=32):
        self.root_path = root_path
        self.batch_size = batch_size
        self.preprocess = Compose([
            Resize((self.img_dim, self.img_dim)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])

def get_data_loader(directory_path, batch_size=32):
    preprocessing_steps = Compose([
        Resize((256, 256)),
        ToTensor()
    ])

    image_dataset = ImageFolder(root=directory_path, transform=preprocessing_steps)
    data_loader = DataLoader(dataset=image_dataset, batch_size=batch_size, shuffle=True)

    return data_loader, image_dataset.classes
