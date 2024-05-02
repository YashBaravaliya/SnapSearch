import streamlit as st
import cv2
import numpy as np

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), channels="BGR")