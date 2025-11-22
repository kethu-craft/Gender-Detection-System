import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw
import cv2
import numpy as np
import gradio as gr

# ------------------------------------------------------
# 🔹 Load the Trained Model
# ------------------------------------------------------
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)  # 3 classes: Male, Female, Animals
model.load_state_dict(torch.load("gender_model.pth", map_location="cpu"))
model.eval()

# ------------------------------------------------------
# 🔹 Image Transformations
# ------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

labels = [ "Female","Male"]

# ------------------------------------------------------
# 🔹 Face Detector (only for human faces)
# ------------------------------------------------------
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ------------------------------------------------------
# 🔹 Prediction Function
# ------------------------------------------------------
def predict_gender(image):
    """
    Detects all human faces and classifies as Male/Female.
    If no faces detected → classifies the entire image (useful for animals).
    """
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    results = []

    # ✅ Case 1: No faces found — classify full image (e.g. animals)
    if len(faces) == 0:
        img_t = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_t)
            _, pred = torch.max(outputs, 1)
            label = labels[pred.item()].capitalize()

        # Color based on prediction
        color = "red" if "animal" in label.lower() else "gray"
        draw.text((10, 10), f"{label}", fill=color)
        return img_draw, f"No human faces detected → Classified as {label}"

    # ✅ Case 2: Faces found — classify each detected face
    for (x, y, w, h) in faces:
        face_crop = image.crop((x, y, x+w, y+h))
        face_t = transform(face_crop).unsqueeze(0)

        with torch.no_grad():
            outputs = model(face_t)
            _, pred = torch.max(outputs, 1)
            label = labels[pred.item()].capitalize()

        # Pick color by class
        if "female" in label.lower():
            color = "pink"
        elif "male" in label.lower():
            color = "blue"
        else:
            color = "red"

        draw.rectangle([(x, y), (x+w, y+h)], outline=color, width=3)
        draw.text((x, y-15), label, fill=color)
        results.append(label)

    return img_draw, ", ".join(results)

# ------------------------------------------------------
# 🔹 Gradio Web UI
# ------------------------------------------------------
interface = gr.Interface(
    fn=predict_gender,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[gr.Image(label="Predicted Image"), gr.Textbox(label="Prediction Result")],
    title="🧑‍🤝‍🧑 Gender & 🐾 Detection Web App",
    description="Upload an image with people or animals. Detects humans as Male/Female and classifies animals if no faces found.",
    allow_flagging="never"
)

# ------------------------------------------------------
# 🔹 Launch App
# ------------------------------------------------------
if __name__ == "__main__":
    interface.launch(share=True)
