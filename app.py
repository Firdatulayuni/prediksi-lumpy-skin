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
        # st.success("Model berhasil dimuat")
        
    except Exception as e:
        st.error("Gagal memuat model:")
        st.exception(e)
else:
    st.error("Folder model tidak ditemukan.")

# Preprocessing Function
def apply_clahe(cv_img):
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return final_img

def preprocess_image(image):
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (224, 224))
    img_clahe = apply_clahe(img_resized)
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB)
    img_scaled = img_rgb.astype(np.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
    img_input = np.expand_dims(img_scaled, axis=0)  # Shape: (1, 224, 224, 3)
    return img_input

# Upload Image
uploaded_file = st.file_uploader("Upload Gambar Sapi", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Gambar yang Diupload")
    image = Image.open(uploaded_file).convert("RGB")
    processed_image = preprocess_image(image)

    if st.button("Prediksi"):
        if infer:
            with st.spinner("Melakukan prediksi..."):
                input_tensor = tf.convert_to_tensor(processed_image, dtype=tf.float32)

                try:
                    # Gunakan nama input dari signature model jika perlu
                    result = infer(input_tensor)

                    # Jika output dict, ambil key-nya
                    output_key = list(result.keys())[0]
                    pred_prob = result[output_key].numpy()[0][0]

                    predicted_label = "Lumpy Skin" if pred_prob > 0.5 else "Normal Skin"
                    # confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob

                    st.subheader("Hasil Prediksi")
                    st.success(f"**Kelas:** `{predicted_label}`")
                    # st.info(f"**Confidence:** `{confidence:.2f}`")

                except Exception as e:
                    st.error("Terjadi kesalahan saat melakukan inferensi:")
                    st.exception(e)
        else:
            st.warning(" Model belum dimuat, prediksi tidak bisa dilakukan.")
