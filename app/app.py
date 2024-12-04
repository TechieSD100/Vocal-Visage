# Import all of the dependencies
import streamlit as st
import os 
import imageio 

import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model
from gif_extraction import load_gif_data

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar:
    st.image("vocal_visage.png")
    st.title('Vocal Visage')
    st.info('Application to test Lip Reading of the videos from the Test DataSet.')
    st.info('Focuses on LipNet based Lip Reading model.')

st.title('Vocal Visage - A New Dimension to Lip Reading')
# Generating a list of options or videos 
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..', 'data', 's1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2:
        video, annotations = load_data(tf.convert_to_tensor(file_path))

        st.info('Original Transcription of the Video:')
        conv1 = tf.strings.reduce_join(num_to_char(annotations)).numpy().decode('utf-8')
        st.text(conv1)

        st.info('Preprocessed GIF that the ML model processes:')
        gif = load_gif_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', gif, fps=10)
        st.image('animation.gif', width=450)

        st.info('Output of the ML model as tokens:')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Predicted Transcription by the model:')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        
