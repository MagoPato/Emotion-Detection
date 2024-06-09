
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import os
import base64

app = Flask(__name__)

# Cargar modelos
script_dir = os.path.dirname(os.path.abspath(__file__))
prototxtPath = os.path.join(script_dir, "face_detector", "deploy.prototxt")
weightsPath = os.path.join(script_dir, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
keras_model_path = os.path.join(script_dir, "modelFEC.h5")

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
emotionModel = load_model(keras_model_path)

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def predict_emotion(frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        if detections[0, 0, i, 2] > 0.4:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (Xi, Yi, Xf, Yf) = box.astype("int")

            if Xi < 0: Xi = 0
            if Yi < 0: Yi = 0

            face = frame[Yi:Yf, Xi:Xf]
            if face.size == 0:
                continue  # Si el rostro está vacío, salta a la siguiente detección
            
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face2 = img_to_array(face)
            face2 = np.expand_dims(face2, axis=0)

            faces.append(face2)
            locs.append((int(Xi), int(Yi), int(Xf), int(Yf)))  # Asegurar que los valores sean int

            pred = emotionModel.predict(face2)
            preds.append(pred[0])

    return (locs, preds)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = base64.b64decode(data['image'])
    npimg = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    locs, preds = predict_emotion(frame)

    results = []
    for (box, pred) in zip(locs, preds):
        result = {
            "box": [int(x) for x in box],  # Convertir a int
            "emotion": classes[np.argmax(pred)],
            "probability": float(max(pred))  # Asegurar que la probabilidad sea float
        }
        print(f"Detected emotion: {result['emotion']} with probability {result['probability']*100:.2f}%")
        results.append(result)

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080,debug=True)
