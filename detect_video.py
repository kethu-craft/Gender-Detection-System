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

# Load video
video_path = r"dataset/video.mp4"
cap = cv2.VideoCapture(video_path)

# Get video info
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video
out = cv2.VideoWriter("people_gender_detected.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

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

                color = (255, 0, 0) if gender == "Male" else (255, 105, 180)
                label = f"{gender} ({conf:.2f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)
    cv2.imshow("Men/Women Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Detection complete! Saved as 'people_gender_detected.mp4'")
