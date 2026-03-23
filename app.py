import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import zipfile
import io

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Shopee Classifier", page_icon="🛍️")

# ------------------ STYLING ------------------
st.markdown("""
<style>
.title {
    text-align:center;
    font-size:36px;
    font-weight:bold;
    color:#ff4b4b;
}
.subtitle {
    text-align:center;
    color:gray;
    margin-bottom:20px;
}
.card {
    padding:10px;
    border-radius:12px;
    background:white;
    box-shadow:0px 3px 10px rgba(0,0,0,0.1);
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ------------------ MODEL ------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('jiya_shopee_final_v2.keras')

model = load_model()

# ------------------ LABELS ------------------
category_names = {
    "00": "Dresses","01": "Long Dresses","02":"Summer Clothes","03":"Winter Clothes",
    "04":"Jeans","05":"Rings","06":"Earrings","07":"Headwear","08":"Purses",
    "09":"Handbags","10":"Mobile Covers","11":"Mobile Phones","12":"Watches",
    "13":"Baby Sippers","14":"Cookers","15":"Coffees","16":"Shoes/Slippers",
    "17":"Heels","18":"Refrigerators","19":"Pendrive","20":"Chair/Stool",
    "21":"Racket","22":"Helmets","23":"Gloves","24":"Wrist Watches","25":"Belts",
    "26":"Earphones/HeadPhones","27":"Toys","28":"Jacket","29":"Pants",
    "30":"Shoes","31":"Snack","32":"Masks","33":"Sanitizers","34":"Skin Products",
    "35":"Perfume","36":"Bathroom supplies","37":"Laptop","38":"Utensils",
    "39":"Home Decor","40":"Showers","41":"Furniture"
}

# ------------------ HEADER ------------------
st.markdown('<div class="title">🛍️ Shopee Product Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload images or a ZIP folder</div>', unsafe_allow_html=True)

# ------------------ UPLOAD OPTIONS ------------------
uploaded_files = st.file_uploader("📁 Upload Images", type=["jpg","jpeg","png"], accept_multiple_files=True)
zip_file = st.file_uploader("🗜️ Or Upload ZIP Folder", type=["zip"])

images = []

# ------------------ HANDLE ZIP ------------------
if zip_file:
    with zipfile.ZipFile(zip_file, "r") as z:
        for file in z.namelist():
            if file.endswith(("jpg","jpeg","png")):
                img = Image.open(io.BytesIO(z.read(file)))
                images.append((file, img))

# ------------------ HANDLE NORMAL FILES ------------------
if uploaded_files:
    for f in uploaded_files:
        img = Image.open(f)
        images.append((f.name, img))

# ------------------ PREDICTION ------------------
if images:
    if st.button("🚀 Classify All"):
        results = []
        progress = st.progress(0)

        cols = st.columns(3)

        for i, (name, image) in enumerate(images):
            # preprocess
            img = image.resize((224,224))
            arr = np.array(img)/255.0
            arr = np.expand_dims(arr, axis=0)

            preds = model.predict(arr, verbose=0)[0]

            # TOP 3
            top3_idx = preds.argsort()[-3:][::-1]

            top_preds = []
            for idx in top3_idx:
                pid = str(idx).zfill(2)
                label = category_names.get(pid, pid)
                conf = preds[idx]*100
                top_preds.append((label, conf))

            results.append({
                "image": name,
                "top1": top_preds[0][0],
                "conf1": top_preds[0][1],
                "top2": top_preds[1][0],
                "conf2": top_preds[1][1],
                "top3": top_preds[2][0],
                "conf3": top_preds[2][1]
            })

            # UI DISPLAY
            with cols[i % 3]:
                st.image(image, use_column_width=True)

                st.markdown('<div class="card">', unsafe_allow_html=True)

                for label, conf in top_preds:
                    st.write(f"**{label}** - {conf:.1f}%")
                    st.progress(int(conf))

                st.markdown('</div>', unsafe_allow_html=True)

            progress.progress((i+1)/len(images))

        # ------------------ RESULTS TABLE ------------------
        df = pd.DataFrame(results)

        st.write("### 📊 Results Table")
        st.dataframe(df)

        # ------------------ SORT OPTION ------------------
        sort = st.selectbox("Sort by", ["None","Top Prediction"])

        if sort == "Top Prediction":
            df = df.sort_values("top1")
            st.dataframe(df)

        # ------------------ DOWNLOAD ------------------
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download CSV",
            csv,
            "predictions.csv",
            "text/csv"
        )
