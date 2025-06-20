import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
data_dir = "/kaggle/input/bloodcell/images/"
train_dir = os.path.join(data_dir, "TRAIN")
val_dir = os.path.join(data_dir, "TEST_SIMPLE")
test_dir = os.path.join(data_dir, "TEST")

# Image augmentations
image_transforms = {
    "TRAIN": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "TEST_SIMPLE": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "TEST": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}
batch_size = 32

datasets_dict = {
    "TRAIN": datasets.ImageFolder(train_dir, transform=image_transforms["TRAIN"]),
    "TEST_SIMPLE": datasets.ImageFolder(val_dir, transform=image_transforms["TEST_SIMPLE"]),
    "TEST": datasets.ImageFolder(test_dir, transform=image_transforms["TEST"]),
}

dataloaders = {
    phase: DataLoader(datasets_dict[phase], batch_size=batch_size, shuffle=(phase == "TRAIN"))
    for phase in ["TRAIN", "TEST_SIMPLE", "TEST"]
}

class_names = datasets_dict["TRAIN"].classes
num_classes = len(class_names)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load VGG19
model = models.vgg19(pretrained=True)

# Freeze all layers
for param in model.features.parameters():
    param.requires_grad = False

# Modify classifier
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
model = model.to(device)
# Load pretrained VGG19
model = models.vgg19(pretrained=True)

# Freeze earlier convolutional layers
for param in model.features.parameters():
    param.requires_grad = False

# Unfreeze last few layers
for param in model.features[-8:].parameters():
    param.requires_grad = True

# Modify the classifier
model.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(4096, num_classes)
)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

best_acc = 0.0
num_epochs = 10
save_path = "/kaggle/working/vgg19_best.pth"

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 30)

    for phase in ["TRAIN", "TEST_SIMPLE"]:
        model.train() if phase == "TRAIN" else model.eval()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.upper()}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "TRAIN"):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == "TRAIN":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(datasets_dict[phase])
        epoch_acc = running_corrects.double() / len(datasets_dict[phase])

        print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Save best model
        if phase == "TEST_SIMPLE" and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), save_path)
            print("✅ Saved new best model")