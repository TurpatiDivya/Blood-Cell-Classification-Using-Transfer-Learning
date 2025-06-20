import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import torch.nn as nn
import matplotlib.pyplot as plt

# === SETUP ===
class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
num_classes = len(class_names)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODEL ===
model = models.vgg19(pretrained=False)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
model.load_state_dict(torch.load("/kaggle/working/vgg19_best.pth"))
model = model.to(device)
model.eval()

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === PREDICTION & DISPLAY FUNCTION ===
def predict_and_show(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        predicted_idx = torch.argmax(probs).item()
        predicted_class = class_names[predicted_idx]
        confidence = probs[predicted_idx].item()

    # Display image with prediction
    plt.imshow(img)
    plt.title(f"{predicted_class} ({confidence * 100:.2f}%)")
    plt.axis("off")
    plt.show()

    return predicted_class, confidence

# === EXAMPLE USAGE ===
predict_and_show("/kaggle/input/bloodcell/images/TEST_SIMPLE/EOSINOPHIL/_9_2814.jpeg")