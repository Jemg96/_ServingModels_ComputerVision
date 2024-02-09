import streamlit as st  # Importar la librería Streamlit
from streamlit_webrtc import webrtc_streamer  # Importar el módulo para trabajar con WebRTC en Streamlit
import av  # Importar la librería PyAV para trabajar con video y audio
import cv2  # Importar OpenCV para procesamiento de imágenes
from PIL import Image  # Importar Pillow para trabajar con imágenes
import numpy as np  # Importar NumPy para operaciones numéricas
import torch  # Importar PyTorch
from transformers import DetrImageProcessor, DetrForObjectDetection  # Importar clases para DETR de la librería transformers

# Inicializar el modelo y procesador de DETR
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# Inicializar la aplicación Streamlit
st.title("Detección de Objetos con DETR")  # Título de la aplicación
st.write("Esta aplicación utiliza DETR para la detección de objetos en tiempo real.")

# Definir la función de callback para el procesamiento de cada frame de video
def callback(frame):
    img = frame.to_ndarray(format="bgr24") # Convertir el frame a un array de NumPy en formato BGR
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Convertir a imagen PIL

    # Procesar la imagen con DETR
    inputs = processor(images=img_pil, return_tensors="pt") # Preparar la entrada para el modelo
    outputs = model(**inputs) # Obtener las salidas del modelo

    # Convertir las salidas a formato COCO API
    target_sizes = torch.tensor([img_pil.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Dibujar los resultados en la imagen
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = box.tolist() # Convertir la caja a lista
        # Dibujar un rectángulo alrededor del objeto detectado
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        # Obtener el nombre de la etiqueta y dibujarla en la imagen
        label_name = model.config.id2label[label.item()]
        cv2.putText(img, f"{label_name} {round(score.item(), 3)}", (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Devolver el frame procesado como un VideoFrame de PyAV
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Iniciar el streamer de WebRTC con la función de callback para el procesamiento del video
webrtc_streamer(key="example", video_frame_callback=callback)
