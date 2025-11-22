# 👥 Gender Detection System

A real-time gender detection system that uses computer vision and deep learning to detect humans in images and classify their gender.

![Gender Detection](https://img.shields.io/badge/Gender-Detection-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)

## 🚀 Features

- **Real-time Person Detection** using YOLOv8
- **Gender Classification** using Caffe model
- **Web Interface** with Flask/Streamlit
- **High Accuracy** with confidence scores
- **User-friendly** drag & drop interface
- **Cross-platform** compatibility

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam (for real-time detection)
- 2GB+ RAM
- 1GB+ free disk space

## 🛠️ Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/Gender-Detection.git
cd Gender-Detection
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

If you don't have requirements.txt, install manually:
```bash
pip install ultralytics opencv-python flask streamlit pillow
```

## 📁 Project Structure
```
Gender-Detection/
│
├── app.py                 # Main application file
├── requirements.txt       # Dependencies list
├── gender_deploy.prototxt # Gender model architecture
├── gender_net.caffemodel  # Gender model weights
├── static/               # Uploaded images folder
├── templates/            # HTML templates (for Flask)
└── README.md            # Project documentation
```

## 🎯 Usage

### Method 1: Using Flask (Web App)
```bash
python app.py
```
Then open: **http://localhost:5000**

### Method 2: Using Streamlit (Interactive App)
```bash
streamlit run app.py
```
Then open: **http://localhost:8501**

### How to Use:
1. **Upload** an image using the file uploader
2. **Click** "Detect Gender" button
3. **View** results with bounding boxes and gender labels
4. **Download** or share the processed image

## 🔧 Models Used

### 1. YOLOv8 (You Only Look Once)
- **Purpose**: Person detection
- **Input**: 640x640 image
- **Output**: Bounding boxes around detected persons
- **Classes**: 80 objects (person = class 0)

### 2. Caffe Gender Model
- **Purpose**: Gender classification
- **Input**: 227x227 face image
- **Output**: Male/Female with confidence score
- **Accuracy**: ~86% on standard datasets

## 🎨 Output Format

- **🟢 Green Box**: Female detection
- **🔵 Blue Box**: Male detection
- **Labels**: Gender with confidence percentage
- **Example**: "Female 95%" or "Male 87%"

## 📊 Performance

- **Detection Speed**: ~50-100ms per image
- **Accuracy**: 85-90% on clear images
- **Supported Formats**: JPG, PNG, JPEG
- **Max Image Size**: 10MB

## ⚠️ Limitations

1. **Image Quality**: Works best with high-resolution images
2. **Face Visibility**: Requires visible front-facing faces
3. **Lighting Conditions**: Performance may vary in low light
4. **Occlusions**: May struggle with partially covered faces
5. **Binary Classification**: Only supports Male/Female classification

## 🔒 Ethical Considerations

- Use responsibly and respect privacy
- Gender classification has inherent limitations
- Results should not be used for critical decisions
- Always obtain proper consent for image processing

## 🐛 Troubleshooting

### Common Issues:

1. **Models not loading**
   - Check if model files exist in project directory
   - Verify file paths in code

2. **No detections**
   - Use images with clear, visible people
   - Ensure good lighting conditions

3. **Installation errors**
   - Use Python 3.8+
   - Create fresh virtual environment
   - Update pip: `python -m pip install --upgrade pip`

4. **Memory issues**
   - Reduce image size before processing
   - Close other applications

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv8
- **OpenCV** team for computer vision tools
- **Caffe** model contributors

## 📞 Support

If you encounter any issues:

1. Check the troubleshooting section
2. Open an issue on GitHub
3. Provide detailed description and error logs

---

## By Kethu! 💖



---
