# models.py
"""
Model manager: load YOLOv8 (ultralytics) for detection and a PyTorch classifier for gender.
Provides a unified interface for detecting persons and classifying gender from face crops.
"""

import torch
import numpy as np
from ultralytics import YOLO
import cv2
from pathlib import Path
import json

class ModelManager:
    def __init__(self, yolo_weights="yolov8n.pt", gender_weights="gender_model.pth", device=None):
        # device: "cpu" or "cuda"
        self.yolo_weights = yolo_weights
        self.gender_weights = gender_weights
        self.yolo = None
        self.gender_model = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.class_map = {0: "Male", 1: "Female"}  # default label mapping
        # lazy load to allow specifying device at inference
        self._yolo_loaded = False
        self._gender_loaded = False

    def load_yolo(self, device=None):
        if self._yolo_loaded and (device is None or device == self.device):
            return
        d = device or self.device
        self.yolo = YOLO(self.yolo_weights)  # ultralytics handles device internally via env param
        # ultralytics YOLO chooses device via YOLO(...).to('cuda') in new versions but
        # we rely on autoselect; can be controlled via `self.yolo.model.to()`.
        try:
            if d == "cuda":
                self.yolo.to("cuda")
            else:
                self.yolo.to("cpu")
        except Exception:
            pass
        self._yolo_loaded = True

    def load_gender(self, device=None):
        d = device or self.device
        map_loc = None if d == "cuda" else torch.device('cpu')
        if self._gender_loaded:
            return
        # Assume gender model is a simple state_dict or entire model saved.
        # Try to load flexibly:
        try:
            # First try loading a full model
            self.gender_model = torch.load(self.gender_weights, map_location=(None if d == "cuda" else "cpu"))
            # If loaded object is a dict (state_dict), user must reconstruct model architecture.
            if isinstance(self.gender_model, dict):
                # try simple linear model guess: fallback to custom small CNN
                # We'll build a lightweight CNN and load state_dict if shapes match
                model = self._simple_cnn()
                try:
                    model.load_state_dict(self.gender_model)
                    model.to(d)
                    model.eval()
                    self.gender_model = model
                except Exception:
                    # not a state_dict matching our small CNN: keep the dict for now
                    pass
            else:
                # If it's a module, move to device
                try:
                    self.gender_model.to(d)
                except Exception:
                    pass
                self.gender_model.eval()
        except Exception as e:
            print("Warning: could not load gender model directly:", e)
            self.gender_model = None
        self._gender_loaded = True

    def _simple_cnn(self):
        # Simple CNN architecture in case the provided .pth is a state_dict matching this.
        import torch.nn as nn
        class SmallCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4))
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64*4*4, 128),
                    nn.ReLU(),
                    nn.Linear(128, 2)
                )
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        return SmallCNN()

    def detect(self, image, device=None, conf=0.25):
        """
        Runs YOLO detection and returns detection results.
        image: numpy array (H,W,3) RGB
        Returns list of detections: each detection is dict with keys:
        - xyxy: [x1,y1,x2,y2]
        - conf: confidence
        - cls: class id (YOLO)
        - name: class name (YOLO)
        """
        self.load_yolo(device=device)
        d = device or self.device
        # ultralytics YOLO expects either image array (BGR/RGB?), works with numpy
        results = self.yolo(image, conf=float(conf), verbose=False)
        # results is a Results object or list
        dets = []
        # Support both single/iterable
        res = results[0] if isinstance(results, (list, tuple)) else results
        boxes = res.boxes
        if boxes is None:
            return dets
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf_score = float(box.conf.cpu().numpy())
            cls_id = int(box.cls.cpu().numpy())
            name = res.names.get(cls_id, str(cls_id))
            dets.append({"xyxy": xyxy, "conf": conf_score, "cls": cls_id, "name": name})
        return dets

    def classify_gender(self, face_img, device=None):
        """
        face_img: numpy image RGB or BGR (H,W,3) uint8
        Returns (label_str, score)
        """
        self.load_gender(device=device)
        d = device or self.device
        if self.gender_model is None:
            # fallback heuristic: predict by simple color/heuristic or return Unknown
            return ("Unknown", 0.0)
        # Preprocess: convert to 224x224, normalize and to tensor
        import torchvision.transforms as T
        from PIL import Image
        img = Image.fromarray(face_img.astype("uint8"))
        transform = T.Compose([
            T.Resize((128,128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        x = transform(img).unsqueeze(0).to(d)
        with torch.no_grad():
            out = self.gender_model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            cls = int(probs.argmax())
            score = float(probs[cls])
            label = self.class_map.get(cls, str(cls))
            return (label, score)
