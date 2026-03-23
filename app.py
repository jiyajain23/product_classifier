import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page config
st.set_page_config(page_title="Shopee Classifier", page_icon="🛍️", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #ff4b4b;
        }
        .subtitle {
            text-align: center;
            font-size: 16px;
            color: gray;
        }
        .card {
            padding: 20px;
            border-radius: 15px;
            background-color: white;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('jiya_shopee_final_v2.keras')

model = load_my_model()

# Categories
category_names = {
    "00": "Dresses", "01": "Long Dresses", "02":"Summer Clothes",
    "03":"Winter Clothes","04":"Jeans","05":"Rings","06":"Earrings",
    "07":"Headwear","08":"Purses","09":"Handbags","10":"Mobile Covers",
    "11":"Mobile Phones","12":"Watches","13":"Baby Sippers",
    "14":"Cookers","15":"Coffees","16":"Shoes/Slippers","17":"Heels",
    "18":"Refrigerators","19":"Pendrive","20":"Chair/Stool",
    "21":"Racket","22":"Helmets","23":"Gloves","24":"Wrist Watches",
    "25":"Belts","26":"Earphones/HeadPhones","27":"Toys","28":"Jacket",
    "29":"Pants","30":"Shoes","31":"Snack","32":"Masks",
    "33":"Sanitizers","34":"Skin Products","35":"Perfume",
    "36":"Bathroom supplies","37":"Laptop","38":"Utensils",
    "39":"Home Decor","40":"Showers","41":"Furniture"
}

# Title
st.markdown('<div class="title">🛍️ Shopee Product Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image and let AI classify it instantly</div>', unsafe_allow_html=True)
st.write("")

# Sidebar
st.sidebar.header("⚙️ Settings")
show_confidence = st.sidebar.checkbox("Show Confidence Score", value=True)

# Upload
uploaded_file = st.file_uploader("📤 Upload a product image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.markdown("### 🧠 Prediction Panel")

        if st.button("🚀 Classify"):
            with st.spinner("Analyzing image..."):
                # Preprocess
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                preds = model.predict(img_array)
                class_idx = np.argmax(preds[0])
                confidence = np.max(preds[0]) * 100

                prediction_id = str(class_idx).zfill(2)
                name = category_names.get(prediction_id, f"Category ID: {prediction_id}")

            # Result Card
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.success(f"🛒 {name}")

            if show_confidence:
                st.progress(int(confidence))
                st.write(f"Confidence: **{confidence:.2f}%**")

            st.markdown('</div>', unsafe_allow_html=True)
