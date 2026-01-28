import streamlit as st
import pandas as pd
import joblib

loaded_model = joblib.load("model.joblib")

st.title("Prediksi Nilai TKA")
st.markdown(
    "Aplikasi Machine Learning regression untuk memprediksi nilai TKA "
    "berdasarkan jam belajar, persentase kehadiran, dan bimbel"
)

jam_belajar_per_hari = st.slider("Jam belajar per hari", 0, 20, 2)
persen_kehadiran = st.slider("Persen kehadiran", 0, 100, 80)

bimbel = st.pills(
    "Mengikuti bimbel?",
    ("ya", "tidak"),
    default="ya"
)

if st.button("Prediksi", type="primary"):
    data_baru = pd.DataFrame({
        "jam_belajar_per_hari": [jam_belajar_per_hari],
        "persen_kehadiran": [persen_kehadiran],
        "bimbel": [bimbel]
    })

    prediksi = loaded_model.predict(data_baru)[0]
    prediksi = max(0, min(100, prediksi))

    st.success(f"Hasil Prediksi: **{prediksi:.0f}**")
    st.balloons()

st.caption("Dibuat dengan ❤️ oleh Ghozy Alfienno M")