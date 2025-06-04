# data_loader.py
print("Script is working!")

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

def get_data_loader(self):
    train_dir = os.path.join(self.data_dir, "train")
    val_dir = os.path.join(self.data_dir, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=self.transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=self.transform)
    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
    
    return train_loader, val_loader
