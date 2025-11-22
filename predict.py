import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# Load the trained model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load("gender_model.pth", map_location="cpu"))
model.eval()

# Define the same transforms used for training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load image for prediction
img_path = r"C:\Users\onimta\Desktop\restart\object_detection\dataset\111.png"  # change this
img = Image.open(img_path).convert("RGB")
img_t = transform(img).unsqueeze(0)

# Predict
with torch.no_grad():
    outputs = model(img_t)
    _, pred = torch.max(outputs, 1)

labels = ["Male", "Female"]
print(f"Predicted Gender: {labels[pred.item()]}")
