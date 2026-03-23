import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Load your saved model (Make sure the file is in the same folder!)
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('jiya_shopee_final_v2.keras')

model = load_my_model()

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Shopee AI Classifier", page_icon="🛍️", layout="wide")

# ---------------------------------------------------------
# 2. ADD YOUR CATEGORY NAMES HERE
# ---------------------------------------------------------
# Replace the text in quotes with your actual category names
category_map = {
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

# 3. Custom Styling
st.markdown("""
    <style>
    .stButton>button { width: 100%; background-color: #ff4b4b; color: white; border-radius: 5px; }
    .prediction-card { padding: 20px; background-color: white; border-radius: 10px; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# 4. Model Loading
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('jiya_shopee_final_v2.keras')

model = load_my_model()

# 5. UI Layout
st.title("🛍️ Shopee Product Intelligence")
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Product Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Product Preview', use_container_width=True)

with col2:
    if uploaded_file:
        st.subheader("Analysis Results")
        
        # Preprocessing
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if st.button('Analyze Product'):
            with st.spinner('AI is thinking...'):
                preds = model.predict(img_array)
                class_idx = np.argmax(preds[0])
                confidence = np.max(preds[0]) * 100
                
                # Get name from our map, fallback to "ID X" if missing
                result_name = category_map.get(class_idx, f"Category {class_idx}")

                st.markdown(f"""
                    <div class="prediction-card">
                        <h2 style='color: #ff4b4b;'>{result_name}</h2>
                        <p>Confidence Level</p>
                        <h1 style='color: #31333F;'>{confidence:.1f}%</h1>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show top 3 matches
                st.write("---")
                st.write("**Top 3 Matches:**")
                top_3 = np.argsort(preds[0])[-3:][::-1]
                for idx in top_3:
                    name = category_map.get(idx, f"ID {idx}")
                    st.write(f"{name}: {preds[0][idx]*100:.1f}%")
                    st.progress(float(preds[0][idx]))
        
        st.success(f"Prediction: **{name}**")
        st.info(f"Confidence: {confidence:.2f}%")
