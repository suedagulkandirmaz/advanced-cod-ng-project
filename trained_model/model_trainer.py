import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

def model_trainer(moel, train_loader, val_loader, num_epochs, device, model_saver):
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), Ir=0.001)

    highest_acc = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        total, true = 0, 0
        for x, y in tqdm(train_data, desc=f"Training Epoch {epoch+1}"):
            x, y = x.model.to(device), y.model.to(device)
            opt.zero_grade()
            pred = model(x) 
            loss = loss_fn(pred,y)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            predicted = torch.argmax(predicted)
            total += y.size(0)
            true += (predicted == y).sum().item()

        acc = check_accuracy(model, val_data, device)
        print(f"Epoch{ep+1} | Loss: {epoch_loss:.3f} | Val Accuracy: {acc:.2f}%")

        if acc > highest_acc:
            highest_acc = acc
            store_model_weights(model, save_path)
            print("New model saved")


