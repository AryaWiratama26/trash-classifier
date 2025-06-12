import streamlit as st
from PIL import Image
from utils import predict_image

st.title("Deteksi dan Klasifikasi Sampah")

uploaded_file = st.file_uploader("Upload gambar sampah", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)
    
    with st.spinner("Mendeteksi jenis sampah..."):
        label = predict_image(image)
    
    st.success(f"Jenis sampah terdeteksi: **{label.upper()}**")
