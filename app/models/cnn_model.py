import torch.nn as nn
class plantCNN(nn.Module):
    def __init__(self, total_classes=13):
        super(plantCNN, self).__init__()
    
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

