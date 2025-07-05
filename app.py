import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

st.set_page_config(page_title="Klasifikasi Lumpy Skin", layout="centered")
st.title("Klasifikasi Penyakit Lumpy Skin pada Sapi")

# Load Model
MODEL_PATH = "best_model_tf2"
model = None
infer = None

if os.path.exists(MODEL_PATH):
    try:
        model = tf.saved_model.load(MODEL_PATH)
        infer = model.signatures["serving_default"]
    except Exception as e:
        st.error("Gagal memuat model:")
        st.exception(e)
else:
    st.error("Folder model tidak ditemukan.")

# CLAHE
def apply_clahe(cv_img):
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return final_img

# Preprocessing
def preprocess_image(image):
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (224, 224))
    img_clahe = apply_clahe(img_resized)
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB)
    img_scaled = img_rgb.astype(np.float32) / 127.5 - 1.0
    img_input = np.expand_dims(img_scaled, axis=0)
    return img_input, img_rgb  # return juga hasil preprocessing dalam format RGB untuk ditampilkan

# Upload Gambar
uploaded_file = st.file_uploader("Upload Gambar Sapi", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Gambar Asli", use_container_width=True)
    image = Image.open(uploaded_file).convert("RGB")

    # Preprocessing
    processed_input, processed_display = preprocess_image(image)

    # Tampilkan hasil preprocessing
    st.subheader("Hasil Preprocessing")
    st.image(processed_display, caption="Gambar Setelah Preprocessing (CLAHE + Resize)", use_container_width=True)

    # Ekstrak label asli dari nama file
    filename = uploaded_file.name  # Contoh: "Normal_Skin_1.jpg"
    name_parts = filename.split("_")
    if len(name_parts) >= 2:
        true_label = f"{name_parts[0]} {name_parts[1]}"
    else:
        true_label = ""

    # Tombol Prediksi
    if st.button("Prediksi"):
        if infer:
            with st.spinner("Melakukan prediksi..."):
                input_tensor = tf.convert_to_tensor(processed_input, dtype=tf.float32)
                try:
                    result = infer(input_tensor)
                    output_key = list(result.keys())[0]
                    pred_prob = result[output_key].numpy()[0][0]
                    predicted_label = "Lumpy Skin" if pred_prob > 0.5 else "Normal Skin"

                    st.subheader("Hasil Prediksi")
                    st.success(f"**Prediksi Model:** `{predicted_label}`")
                    st.info(f"**Probabilitas Lumpy:** `{pred_prob:.4f}`")

                    if true_label:
                        st.subheader("Label Asli (Ground Truth)")
                        st.code(true_label)

                except Exception as e:
                    st.error("Terjadi kesalahan saat melakukan inferensi:")
                    st.exception(e)
        else:
            st.warning("Model belum dimuat, prediksi tidak bisa dilakukan.")
