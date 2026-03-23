import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Load your saved model (Make sure the file is in the same folder!)
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('jiya_shopee_final_v2.keras')

model = load_my_model()

category_names = {
    "00": "Dresses",
    "01": "Long Dresses",
    "02":"Summer Clothes",
    "03":"Winter Clothes",
    "04":"Jeans",
    "05":"Rings",
    "06":"Earrings",
    "07":"Headwear",
    "08":"Purses",
    "09":"Handbags",
    "10":"Mobile Covers",
    "11":"Mobile Phones",
    "12":"Watches",
    "13":"Baby Sippers",
    "14":"Cookers",
    "15":"Coffees",
    "16":"Shoes/Slippers",
    "17":"Heels",
    "18":"Refridgerators",
    "19": "Pendrive",
    "20":"Chair/Stool",
    "21":"Racket",
    "22":"Helmets",
    "23":"Gloves",
    "24":"Wrist Watches",
    "25":"Belts",
    "26":"Earphones/HeadPhones",
    "27":"Toys",
    "28":"Jacket",
    "29":"Pants",
    "30":"Shoes",
    "31":"Snack",
    "32":"Masks",
    "33":"Sanitizers",
    "34":"Skin Products",
    "35":"Perfume",
    "36":"Bathroom supplies",
    "37":"Laptop",
    "38":"Utensils",
    "39":"Home Decor",
    "40":"Showers",
    "41":"Furniture"
}

st.title("🛍️ Shopee Product Classifier")
st.write("Upload a product image to identify its category.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    if st.button('Classify'):
        preds = model.predict(img_array)
        class_idx = np.argmax(preds[0])
        
        # We need the labels list from your training code
        # For now, let's assume class_idx maps to the ID string
        prediction_id = str(class_idx).zfill(2) # Matches "01", "19", etc.
        name = category_names.get(prediction_id, f"Category ID: {prediction_id}")
        confidence = np.max(preds[0]) * 100
        
        st.success(f"Prediction: **{name}**")
        st.info(f"Confidence: {confidence:.2f}%")
