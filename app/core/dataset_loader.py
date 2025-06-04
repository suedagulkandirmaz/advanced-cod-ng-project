# data_loader.py

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader

def get_data_loader(directory_path, batch_size=32):
    preprocessing_steps = Compose([
        Resize((256, 256)),
        ToTensor()
    ])

    image_dataset = ImageFolder(root=directory_path, transform=preprocessing_steps)
    data_loader = DataLoader(dataset=image_dataset, batch_size=batch_size, shuffle=True)

    return data_loader, image_dataset.classes
