from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
import cv2, os

app = Flask(__name__)

# Load models once
model = YOLO("yolov8n.pt")
gender_net = cv2.dnn.readNetFromCaffe(
    "gender_deploy.prototxt",
    "gender_net.caffemodel"
)
GENDER_LIST = ["Male", "Female"]

UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        # Save uploaded image
        input_path = os.path.join(UPLOAD_FOLDER, "input.jpg")
        file.save(input_path)

        # Run detection
        frame = cv2.imread(input_path)
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

                    # Color and label
                    color = (255, 0, 0) if gender == "Male" else (255, 105, 180)
                    label = f"{gender} ({conf:.2f})"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        output_path = os.path.join(UPLOAD_FOLDER, "output.jpg")
        cv2.imwrite(output_path, frame)

        return render_template("index.html", result_image="output.jpg")

    return render_template("index.html", result_image=None)

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
