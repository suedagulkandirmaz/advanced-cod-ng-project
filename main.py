import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from app.models.cnn_model import cnn_model
from app.core.predictor import model_trainer
import os

def main():
    data_dir = 'PlantVillage'
    num_classes = 13
    batch_size = 32
    num_epoch = 10
    device = torch.device('cuda' if torch.cuda_is.available() else 'cpu')
    model_save_path = 'trained_model/plant_disease_model.pt'

    transform = transform.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.IamageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    model = PlantDiseaseCNN(num_classes=num_classes)

    model_trainer(model, train_loader, val_loader, num_epoch, device, model_save_path)

if __name__=="__main__":
    main()