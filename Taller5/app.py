import streamlit as st

# Título de la aplicación
st.title('Simple Streamlit App')

# Crear un campo de entrada de texto
input_text = st.text_input('Ingrese algún texto')

# Crear un botón
if st.button('Mostrar Imagen'):
    ###############
    ####LOGICA#####
    ###############
    # Si se hace clic en el botón, mostrar la imagen
    st.image('https://pub-3626123a908346a7a8be8d9295f44e26.r2.dev/generations/0-430cb9b2-bc0a-4a23-8ae3-b7290fecea1a.png',
             caption='Imagen Mostrada')