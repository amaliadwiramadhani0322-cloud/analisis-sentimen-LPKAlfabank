import streamlit as st
import joblib
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Analisis Sentimen ALFABANK",
    layout="centered"
)

# CUSTOM CSS
st.markdown("""
<style>

/* Background */
.stApp {
    background: #f1f5f9;
    font-family: 'Segoe UI', sans-serif;
}

/* Main Container */
.block-container {
    max-width: 760px;
    padding-top: 3rem;
    padding-bottom: 2rem;
}

/* Title */
.title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.2;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 24px;
    font-weight: 600;
    color: #334155;
    margin-bottom: 10px;
}

/* Naikin area input */
.stTextArea {
    margin-top: -8px;
}

/* Label */
.stTextArea label {
    font-size: 17px;
    font-weight: 600;
    color: #334155;
}

/* Text Area */
.stTextArea textarea {
    border-radius: 18px !important;
    border: 2px solid #dbe4ee !important;
    padding: 16px !important;
    font-size: 17px !important;
    background-color: white !important;
    min-height: 90px !important;
}

.stTextArea textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
}

/* Button */
.stButton {
    display: flex;
    justify-content: center;
    margin-top: 12px;
}

.stButton button {
    width: 100%;
    border-radius: 16px;
    height: 58px;
    background: #2563eb;
    color: white;
    font-size: 18px;
    font-weight: 600;
    border: none;
    transition: 0.3s ease;
}

.stButton button:hover {
    background: #1d4ed8;
    transform: translateY(-2px);
}

/* Result */
.result-box {
    margin-top: 28px;
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    font-size: 22px;
    font-weight: 700;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# LOAD MODEL
try:
    model = joblib.load("svm_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    st.error(f"Gagal load model: {e}")
    st.stop()


# PREPROCESSING
stemmer = StemmerFactory().create_stemmer()
factory = StopWordRemoverFactory()

stopwords = set(factory.get_stop_words())
negations = {"tidak", "bukan", "jangan", "belum", "kurang"}
stopwords = stopwords - negations

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords]
    stemmed = [stemmer.stem(word) for word in tokens]

    return " ".join(stemmed)


# HEADER
st.markdown("""
<div class="title">
    Analisis Sentimen Ulasan Peserta ALFABANK
</div>

<div class="subtitle">
    Masukkan Ulasan Peserta
</div>
""", unsafe_allow_html=True)

# INPUT
user_input = st.text_area(
    "Input Ulasan",
    placeholder="Masukkan ulasan peserta pelatihan..."
)

# BUTTON
if st.button("Prediksi Sentimen"):

    if user_input.strip() == "":
        st.warning("Silakan masukkan ulasan terlebih dahulu.")

    else:
        clean_text = preprocess_text(user_input)
        vector = tfidf.transform([clean_text])
        prediction = model.predict(vector)[0]

        if prediction == "positif":
            st.markdown("""
            <div class="result-box" style="background:#22c55e;">
                Hasil Prediksi Sentimen: POSITIF
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="result-box" style="background:#ef4444;">
                Hasil Prediksi Sentimen: NEGATIF
            </div>
            """, unsafe_allow_html=True)
