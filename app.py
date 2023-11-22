from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import streamlit as st
import cv2
import io
import base64
from PIL import Image  # Import the Image module from the PIL library

def crop_image(image, x, y, width, height):
    cropped_image = image[y:y+height, x:x+width]
    return cropped_image


model = load_model('Model.h5')
st.title("Prediction of Pocs")
file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])

def highlight_non_black_regions(uploaded_image):
    try:
        img = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img, 10, 30, cv2.THRESH_BINARY)
        highlighted_img = uploaded_image.copy()
        highlighted_img[mask == 0] = 255 # White color for non-black regions
        return highlighted_img
    except Exception as e:
        st.error(f"An error occurred while processing the image: {str(e)}")
        return None

# Initialize width and height variables
width, height = 0, 0

if file is not None:
    image_data = file.read()
    ss = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    x = 91
    y = 83
    width = ss.shape[1]
    height = ss.shape[0]
    cropped_img = crop_image(ss, x, y, width, height)
    st.image(ss, caption="Original Image",use_column_width=False, width=300)
    st.subheader("Adjust cropping dimensions:")
    x = st.slider("X Coordinate", min_value=0, max_value=width - 1, value=x)
    y = st.slider("Y Coordinate", min_value=0, max_value=height - 1, value=y)
    width = st.slider("Width", min_value=1, max_value=width - x, value=width)
    height = st.slider("Height", min_value=1, max_value=height - y, value=height)
    cropped_img = crop_image(ss, x, y, width, height)
else:
    st.warning("Please upload an image for prediction")

# Streamlit button to trigger the prediction
if st.button("Predict"):
    if file is not None:
        processed_image = highlight_non_black_regions(cropped_img)
        
        # Convert the processed image (NumPy array) to BytesIO
        processed_image_io = io.BytesIO()
        processed_image_pil = Image.fromarray(processed_image)
        processed_image_pil.save(processed_image_io, format='PNG')
        
        # Load the image using image.load_img
        img = image.load_img(processed_image_io, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        pred = model.predict(x)
        class_labels = ['infected', 'notinfected']
        class_idx = np.argmax(pred)
        
        if processed_image is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(ss, caption="Original Image", use_column_width=False, width=300)  # Adjust width
                st.caption("Original Image")
            
            with col2:
                st.image(processed_image, caption="Cropped & Highlighted Image", use_column_width=False, width=300)  # Adjust width
                st.caption("Cropped & Highlighted Image")
            
            if class_idx == 0:
                st.error("Infected with Pocs")
            else:
                st.success("Not Infected by Pocs")
    else:
        st.warning("Please upload an image for prediction")
