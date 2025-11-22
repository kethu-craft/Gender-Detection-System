from ultralytics import YOLO
import cv2

# Load YOLOv8 person detector
model = YOLO("yolov8n.pt")

# Load gender classification model (Caffe)
gender_net = cv2.dnn.readNetFromCaffe(
    "gender_deploy.prototxt",
    "gender_net.caffemodel"
)

GENDER_LIST = ["Male", "Female"]

# ---- Upload or set image path ----
image_path = r"dataset/111.png"  # 👈 change this to your image path

# Load image
frame = cv2.imread(image_path)
if frame is None:
    raise FileNotFoundError("❌ Image not found! Please check your image_path.")

# Run YOLO detection
results = model(frame, verbose=False)

# Loop over detections
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        if model.names[cls] == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Crop person region
            person_img = frame[y1:y2, x1:x2].copy()

            try:
                blob = cv2.dnn.blobFromImage(person_img, 1.0, (227, 227),
                                             (78.4263377603, 87.7689143744, 114.895847746),
                                             swapRB=False)
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = GENDER_LIST[gender_preds[0].argmax()]
            except:
                gender = "Unknown"

            # Color and label
            color = (255, 0, 0) if gender == "Male" else (255, 105, 180)
            label = f"{gender} ({conf:.2f})"

            # Draw on image
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ---- Save and show output ----
output_path = "111.png"
cv2.imwrite(output_path, frame)
print(f"✅ Detection complete! Saved as '{output_path}'")

cv2.imshow("Detected Image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
