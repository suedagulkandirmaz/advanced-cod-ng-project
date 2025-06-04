import torch.nn as nn
class plantCNN(nn.modele):
    def __init__(self, total_classes=13):
        super(plantCNN, self).self__init__
    
#FEATURES

        self.conv_layers = nn.Sequential(
            nn.conv2d(in_channels=3, out_channels=20, kernel_size=3, padding=1),
            nn.LeakyRelu(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(20, 40, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(40, 80, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
)
        
# Classifier layers

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*64*32, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.3),
            nn.Linear(256, total_classes)
        )
    
def forward(self, x):
    x = self.conv_layers(x)
    x = self.classifier(x)
    return x









































def plot_training_graph(history, save=False, file_name="training_plot.png"):
    plt.style.use("ggplot")

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green')
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='red')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save:
        plt.savefig(file_name)
        print(f"Plot saved as {file_name}")
    else:
        plt.show()


if __name__ == "__main__":
    dataset_path = 'PlantVillage'  
    train_data, val_data = prepare_data(dataset_path)

    num_classes = len(train_data.class_indices)
    model = create_CNN_model(num_classes=num_classes)

    history = model.fit(
        train_data,
        epochs=10,
        validation_data=val_data
    )

    
    plot_training_graph(history, save=True)

    
    model.save("plant_disease_model.h5")


    with open("class_indices.json", "w") as f:
        json.dump(train_data.class_indices, f)

    print("Model and class saved.")
