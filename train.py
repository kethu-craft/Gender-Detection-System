import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import json
import os

# ================================
# 🔧 Image Transformations
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================================
# 📂 Load Datasets
# ================================
train_dir = r"C:\Users\onimta\Desktop\restart\object_detection\train"
val_dir = r"C:\Users\onimta\Desktop\restart\object_detection\val"

train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

# Save class names to JSON (to use in prediction)
class_to_idx = train_data.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
with open("class_labels.json", "w") as f:
    json.dump(idx_to_class, f)
print("✅ Saved class label mapping:", idx_to_class)

# ================================
# 🧩 Load Pretrained MobileNetV2
# ================================
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
num_classes = len(train_data.classes)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ================================
# ⚙️ Loss & Optimizer
# ================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# ================================
# 🚀 Training Loop
# ================================
for epoch in range(5):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/5] - Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "gender_model.pth")
print("✅ Training complete! Model saved as gender_model.pth")
