import streamlit as st
import pandas as pd

st.title("Heart Disease Prediction")

st.write("Upload test CSV file (small dataset only)")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully")
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())
else:
    st.info("No file uploaded. Please upload a test CSV file.")