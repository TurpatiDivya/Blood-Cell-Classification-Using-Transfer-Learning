
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import cv2
import base64

app = Flask(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load your trained VGG19 model
import torchvision.models as models

# Make sure to adapt this to your fine-tuned VGG19 class definition
model = models.vgg19(pretrained=False)
model.classifier[6] = torch.nn.Linear(4096, 4)   # Assuming you have 4 classes
model.load_state_dict(torch.load(r"C:\Users\IIITDMK-EC\Videos\blood\vgg19_best.pth", map_location=device))
model.to(device)
model.eval()

class_labels = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

def predict_image_class(image_path, model):
    # PyTorch Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class_label = class_labels[predicted.item()]

    # For display, convert PIL image to numpy rgb
    img_rgb = np.array(image)
    return predicted_class_label, img_rgb

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            # Save to static
            file_path = os.path.join("static", file.filename)
            file.save(file_path)
            predicted_class_label, img_rgb = predict_image_class(file_path, model)

            # Convert image to string for displaying in HTML
            _, img_encoded = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            img_str = base64.b64encode(img_encoded).decode('utf-8')

            return render_template("result.html", class_label=predicted_class_label, img_data=img_str)
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)