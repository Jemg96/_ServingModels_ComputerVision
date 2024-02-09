import streamlit as st  # Importar la librería Streamlit
from streamlit_webrtc import webrtc_streamer  # Importar el módulo para trabajar con WebRTC en Streamlit
import av  # Importar la librería PyAV para trabajar con video y audio
import cv2  # Importar OpenCV para procesamiento de imágenes

# Establecer el título de la aplicación web en Streamlit
st.title("Computer Vision Streamlit app")
# Mostrar un mensaje de texto en la aplicación
st.write("Hello, world")

# Definir una función de callback que procesará cada frame del video
def callback(frame):
    # Convertir el frame a un array de NumPy en formato BGR (usado por OpenCV)
    img = frame.to_ndarray(format="bgr24")

    # Aplicar el detector de bordes Canny a la imagen y luego convertir a color para que sea compatible con el formato de video
    img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

    # Devolver el frame procesado como un VideoFrame de PyAV
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Iniciar un streamer de WebRTC en la aplicación Streamlit.
# 'key' es un identificador único para este streamer.
# 'video_frame_callback' es la función que se llama con cada frame del video para su procesamiento.
webrtc_streamer(key="example", video_frame_callback=callback)