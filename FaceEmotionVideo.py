## Detección de emociones en tiempo real ##
## SISTEMAS INTELIGENTES ##
## https://www.youtube.com/channel/UCr_dJOULDvSXMHA1PSHy2rg
## David Revelo Luna

# Importar librerías
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
import time
import logging
import os

# Configurar logging
logging.basicConfig(filename='face_emotion.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# Variables para calcular FPS
time_actualframe = 0
time_prevframe = 0

# Tipos de emociones del detector
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

try:
    # Obtener el directorio del script actual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construir rutas relativas a los archivos de modelo
    prototxtPath = os.path.join(script_dir, "face_detector", "deploy.prototxt")
    weightsPath = os.path.join(script_dir, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
    keras_model_path = os.path.join(script_dir, "modelFEC.h5")

    # Verificar si los archivos existen
    if not os.path.isfile(prototxtPath):
        logging.error(f"Archivo Prototxt no encontrado: {prototxtPath}")
        raise FileNotFoundError(f"Archivo Prototxt no encontrado: {prototxtPath}")
    if not os.path.isfile(weightsPath):
        logging.error(f"Archivo Weights no encontrado: {weightsPath}")
        raise FileNotFoundError(f"Archivo Weights no encontrado: {weightsPath}")
    if not os.path.isfile(keras_model_path):
        logging.error(f"Archivo del modelo Keras no encontrado: {keras_model_path}")
        raise FileNotFoundError(f"Archivo del modelo Keras no encontrado: {keras_model_path}")

    # Cargar los modelos
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    logging.info("Modelo de red cargado con éxito")
    
    emotionModel = load_model(keras_model_path)
    logging.info("Modelo Keras cargado con éxito")

except Exception as e:
    logging.exception("Ocurrió un error al cargar los modelos")
    raise

# Se crea la captura de video
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Toma la imagen, los modelos de detección de rostros y mascarillas
# Retorna las localizaciones de los rostros y las predicciones de emociones de cada rostro
def predict_emotion(frame, faceNet, emotionModel):
    # Construye un blob de la imagen
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # Realiza las detecciones de rostros a partir de la imagen
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Listas para guardar rostros, ubicaciones y predicciones
    faces = []
    locs = []
    preds = []

    # Recorre cada una de las detecciones
    for i in range(0, detections.shape[2]):

        # Fija un umbral para determinar que la detección es confiable
        # Tomando la probabilidad asociada en la detección
        if detections[0, 0, i, 2] > 0.4:
            # Toma el bounding box de la detección escalado
            # de acuerdo a las dimensiones de la imagen
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (Xi, Yi, Xf, Yf) = box.astype("int")

            # Valida las dimensiones del bounding box
            if Xi < 0: Xi = 0
            if Yi < 0: Yi = 0

            # Se extrae el rostro y se convierte BGR a GRAY
            # Finalmente se escala a 224x244
            face = frame[Yi:Yf, Xi:Xf]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face2 = img_to_array(face)
            face2 = np.expand_dims(face2, axis=0)

            # Se agrega los rostros y las localizaciones a las listas
            faces.append(face2)
            locs.append((Xi, Yi, Xf, Yf))

            pred = emotionModel.predict(face2)
            preds.append(pred[0])

    return (locs, preds)

while True:
    try:
        # Se toma un frame de la cámara y se redimensiona
        ret, frame = cam.read()
        if not ret:
            logging.error("No se pudo capturar el frame de la cámara")
            break

        frame = imutils.resize(frame, width=640)

        (locs, preds) = predict_emotion(frame, faceNet, emotionModel)

        # Para cada hallazgo se dibuja en la imagen el bounding box y la clase
        for (box, pred) in zip(locs, preds):
            (Xi, Yi, Xf, Yf) = box
            (angry, disgust, fear, happy, neutral, sad, surprise) = pred

            # Se agrega la probabilidad en el label de la imagen
            label = "{}: {:.0f}%".format(classes[np.argmax(pred)], max(angry, disgust, fear, happy, neutral, sad, surprise) * 100)

            cv2.rectangle(frame, (Xi, Yi-40), (Xf, Yi), (255, 0, 0), -1)
            cv2.putText(frame, label, (Xi+5, Yi-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (255, 0, 0), 3)

        time_actualframe = time.time()
        fps = 1 / (time_actualframe - time_prevframe)
        time_prevframe = time_actualframe

        cv2.putText(frame, str(int(fps)) + " FPS", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        logging.exception("Ocurrió un error durante la captura de video")

cv2.destroyAllWindows()
cam.release()
