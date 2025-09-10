
# 😟 Stress Detection using Facial Emotion Recognition (FER)

This project detects stress-related emotions from facial expressions using a deep learning model. A simple web interface is built with Streamlit where users can upload an image, and the model predicts the emotion on the detected face.

## 📌 Features

- Upload an image and detect emotions using a trained CNN model.
- Facial emotion recognition using OpenCV and Keras.
- Seven emotion classes: `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`.
- Visual feedback with bounding box and predicted emotion.

## 🖥️ Demo

Upload any clear frontal image containing a human face, and the app will highlight the detected face with the predicted emotion label.

> ✅ Useful for understanding mood/stress indicators in workplace scenarios.

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – For building the web interface
- **OpenCV** – Face detection and image processing
- **Keras (TensorFlow backend)** – Emotion classification model
- **NumPy** – Array manipulations

## 🧠 Emotion Recognition Model

- Trained on the **FER2013** dataset
- CNN-based architecture
- Model files:
  - `fer.json` – Model architecture in JSON format
  - `fer.h5` – Trained weights
  - `haarcascade_frontalface_default.xml` – Face detection model

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Prakash1210/stress-detection.git
cd stress-detection
````

 2. Install dependencies

Create a virtual environment (optional but recommended), then install packages:

```bash
pip install -r requirements.txt
```

Sample `requirements.txt`:

```txt
streamlit
opencv-python
numpy
keras
tensorflow
```
 3. Run the Streamlit app

```bash
streamlit run prakash.py
```

4. Upload an image and view the results

Use images with clearly visible faces for best results.

## 📂 Project Structure

```
stress-detection-fer/
│
├── fer.json                         # Model architecture
├── fer.h5                           # Trained model weights
├── haarcascade_frontalface_default.xml  # OpenCV face detection model
├── app.py                           # Main Streamlit application
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

 📸 Sample Output

<img src="sample_output.png" alt="Emotion detection output" width="500"/>

 🔮 Future Improvements

* Live webcam-based real-time emotion detection
* Time-series emotion analysis to detect prolonged stress
* Integration with wellness dashboards or HR monitoring tools

## 👨‍💻 Author

* **Prakash B** – Web Developer | Software Developer 

## 📄 License

This project is open-source and available under the MIT License.

---

### 💬 *"Understanding emotions is the first step toward improving mental well-being."*

