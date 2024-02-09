import streamlit as st
import requests  
import json  

# Título de la aplicación
st.title('Simple Streamlit App')

# Crear un campo de entrada de texto
input_text = st.text_input('Ingrese algún texto')

# Crear un botón
if st.button('Mostrar Imagen'):
    url =  "https://stablediffusionapi.com/api/v4/dreambooth"  

    payload = json.dumps({  
    "key":  "TOKEN_ID",  
    "model_id":  "juggernaut-xl-v5",  
    "prompt":  input_text,  
    "negative_prompt":  "painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime",  
    "width":  "512",  
    "height":  "512",  
    "samples":  "1",  
    "num_inference_steps":  "30",  
    "safety_checker":  "no",  
    "enhance_prompt":  "yes",  
    "seed":  None,  
    "guidance_scale":  7.5,  
    "multi_lingual":  "no",  
    "panorama":  "no",  
    "self_attention":  "no",  
    "upscale":  "no",  
    "embeddings":  "embeddings_model_id",  
    "lora":  "lora_model_id",  
    "webhook":  None,  
    "track_id":  None  
    })  

    headers =  {  
    'Content-Type':  'application/json'  
    }  
    
    response = requests.request("POST", url, headers=headers, data=payload) 

    image_link = response.json()['output'][0]

    # Si se hace clic en el botón, mostrar la imagen
    st.image(image_link,
             caption='Imagen Mostrada')