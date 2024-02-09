# Descargar YoloV3.weights: https://www.kaggle.com/datasets/shivam316/yolov3-weights

import streamlit as st  # Importar la librería Streamlit
from streamlit_webrtc import webrtc_streamer  # Importar el módulo para trabajar con WebRTC en Streamlit
import av  # Importar la librería PyAV para trabajar con video y audio
import cv2  # Importar OpenCV para procesamiento de imágenes
import numpy as np  # Importar NumPy para operaciones numéricas

# Cargar YOLOv3
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg") # Cargar pesos y configuración de YOLO
layer_names = net.getLayerNames() # Obtener los nombres de todas las capas en la red
out_layer_indices = net.getUnconnectedOutLayers() # Obtener los índices de las capas de salida

# Asegurarse de que los índices de las capas de salida sean una lista plana
if isinstance(out_layer_indices, np.ndarray):
    out_layer_indices = out_layer_indices.flatten()

# Obtener los nombres de las capas de salida
output_layers = [layer_names[i - 1] for i in out_layer_indices]

# Cargar los nombres de las clases
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()] # Leer los nombres de las clases

# Initialize Streamlit app
st.title("Detección de Objetos con YOLOv3")  # Título de la aplicación
st.write("Esta aplicación utiliza YOLOv3 para la detección de objetos en tiempo real.")

# Definir la función de callback para el procesamiento de cada frame de video
def callback(frame):
    img = frame.to_ndarray(format="bgr24") # Convertir el frame a un array de NumPy en formato BGR

    # Detección de objetos
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False) # Preprocesar la imagen
    net.setInput(blob) # Establecer la imagen como entrada de la red
    outs = net.forward(output_layers) # Realizar la detección

    # Procesar las detecciones
    for out in outs:
        for detection in out:
            scores = detection[5:] # Obtener las puntuaciones de las clases
            class_id = np.argmax(scores) # Obtener el ID de la clase con la mayor puntuación
            confidence = scores[class_id] # Obtener la confianza de la detección
            if confidence > 0.98: # Filtrar detecciones por confianza
                # Calcular las coordenadas del rectángulo de la detección
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Coordenadas del rectángulo
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Dibujar rectángulo y texto en la imagen
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = str(classes[class_id])
                cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Devolver el frame procesado como un VideoFrame de PyAV
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Iniciar el streamer de WebRTC con la función de callback para el procesamiento del video
webrtc_streamer(key="example", video_frame_callback=callback)


