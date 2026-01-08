import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset
from PIL import Image

# =========================================================
# DATASET
# =========================================================
class PetDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.filepaths[idx]).convert("RGB")
            if self.transform:
                image = self.transform(image)
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        except Exception:
            return torch.zeros((3, 224, 224)), torch.tensor(0.0)


# =========================================================
# CUSTOM ADVANCED CNN (NO SIGMOID)
# =========================================================
class AdvancedCNN(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        class ResidualBlock(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.block = nn.Sequential(
                    conv_block(c, c),
                    conv_block(c, c)
                )

            def forward(self, x):
                return x + self.block(x)

        self.features = nn.Sequential(
            conv_block(3, 32),
            nn.MaxPool2d(2),
            ResidualBlock(32),

            conv_block(32, 64),
            nn.MaxPool2d(2),
            ResidualBlock(64),

            conv_block(64, 128),
            nn.MaxPool2d(2),
            ResidualBlock(128),

            conv_block(128, 256),
            nn.MaxPool2d(2),
            ResidualBlock(256),

            conv_block(256, 512)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)   # LOGITS OUTPUT
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# =========================================================
# MODEL FACTORY
# =========================================================
def get_model(model_name, device):
    print(f"Initializing model: {model_name}")

    if model_name == "custom_cnn":
        model = AdvancedCNN()

    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        for p in model.features.parameters():
            p.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, 1)
        )

    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for p in model.parameters():
            p.requires_grad = False
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for p in model.features.parameters():
            p.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 1)
        )

    else:
        raise ValueError("Unknown model name")

    return model.to(device)
