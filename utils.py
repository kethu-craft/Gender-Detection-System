# utils.py
"""
Image/video processing utilities: drawing boxes, cropping faces, video frame loop.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from models import ModelManager
import math
import tempfile
import os

FONT = cv2.FONT_HERSHEY_SIMPLEX

def xyxy_to_int(xyxy):
    return [int(max(0, round(v))) for v in xyxy]

def crop_box(img, xyxy, expand_px=0):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(max(0, round(v))) for v in xyxy]
    x1 = max(0, x1 - expand_px)
    y1 = max(0, y1 - expand_px)
    x2 = min(w - 1, x2 + expand_px)
    y2 = min(h - 1, y2 + expand_px)
    return img[y1:y2, x1:x2]

def draw_label(img, text, xyxy, color=(0,255,0), thickness=2):
    x1, y1, x2, y2 = xyxy_to_int(xyxy)
    # draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness)
    # label background
    (w_text, h_text), baseline = cv2.getTextSize(text, FONT, 0.6, 1)
    # top-left for text background
    cv2.rectangle(img, (x1, y1 - h_text - 8), (x1 + w_text + 8, y1), color, -1)
    # put text
    cv2.putText(img, text, (x1 + 4, y1 - 6), FONT, 0.6, (255,255,255), 1, cv2.LINE_AA)

def annotate_image(image, model_mgr: ModelManager, device="cpu", conf_thres=0.25):
    """
    image: numpy array RGB (gradio gives RGB)
    returns annotated RGB image
    """
    # convert RGB -> BGR for opencv drawing if needed, but we'll keep RGB for consistency
    img = image.copy()
    # Ensure model is loaded
    model_mgr.load_yolo(device=device)
    model_mgr.load_gender(device=device)
    # Run detection (YOLO expects either BGR or RGB; ultralytics can handle RGB images)
    results = model_mgr.detect(img, device=device, conf=conf_thres)
    # Iterate detections, filter persons if name indicates 'person'
    for det in results:
        name = det.get("name", "")
        if name.lower() not in ("person", "people", "personnel", "human") and det.get("cls", None) is not None:
            # If dataset uses person class id 0 usually; but we allow classification for all detected objects as candidate person
            pass
        xyxy = det["xyxy"]
        conf = det["conf"]
        # crop face region: YOLO detects full person; better to use face detector but we'll crop upper part as face estimate.
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        h = y2 - y1
        # assume face region roughly upper 35% of the detected person bounding box
        fh = max(8, int(h * 0.35))
        fx1, fy1 = x1, y1
        fx2, fy2 = x2, y1 + fh
        # keep inside image
        fx1 = max(0, fx1); fy1 = max(0, fy1)
        fx2 = min(img.shape[1]-1, fx2); fy2 = min(img.shape[0]-1, fy2)
        face_crop = img[fy1:fy2, fx1:fx2]
        label = "Unknown"
        score = 0.0
        if face_crop is not None and face_crop.size != 0:
            label, score = model_mgr.classify_gender(face_crop, device=device)
        display_text = f"{label} {score:.2f}"
        draw_label(img, display_text, xyxy, color=(0,165,255))
    # return RGB (gradio expects RGB)
    return img

def process_video_file(in_path: str, out_path: str, model_mgr: ModelManager, device="cpu", conf_thres=0.25):
    """
    Read video frames, annotate each, write output mp4 using OpenCV
    """
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {in_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w,h))
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # frame is BGR from OpenCV; convert to RGB for model utils
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated = annotate_image(rgb, model_mgr, device=device, conf_thres=conf_thres)
            # convert back to BGR for writing
            bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            out.write(bgr)
            frame_idx += 1
    finally:
        cap.release()
        out.release()
    return out_path
