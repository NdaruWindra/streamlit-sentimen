import streamlit as st
import pickle
import os

# Menggunakan os.path.join untuk memastikan jalur yang benar
current_dir = os.path.dirname(os.path.abspath(__file__))
vectorizer_path = os.path.join(current_dir, "tfidf.pkl")
model_path = os.path.join(current_dir, "naive_bayes_model.pkl")

# Load model and vectorizer
with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

def sentimen_label(prediction):
    if prediction == 0:
        return 'Negatif'
    elif prediction == 1:
        return 'Positif'

def predict_sentiment(text):
    vector = vectorizer.transform([text])
    predicted_sentimen = model.predict(vector)[0]
    return sentimen_label(predicted_sentimen)

# Streamlit UI
st.title("Analisis Sentimen Mobil Listrik")

text_input = st.text_area("Masukkan teks yang ingin diprediksi", height=200)

if st.button("Prediksi"):
    if text_input:
        prediction = predict_sentiment(text_input)
        st.write(f"Hasil Prediksi: **{prediction}**")
    else:
        st.write("Silakan masukkan teks untuk diprediksi.")
