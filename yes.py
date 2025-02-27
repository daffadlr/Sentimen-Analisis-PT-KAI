import streamlit as st
import pandas as pd
import re
import plotly.express as px
from transformers import pipeline
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load IndoBERT Sentiment Analysis Model
sentiment_pipeline = pipeline("text-classification", model="w11wo/indonesian-roberta-base-sentiment-classifier")

# Fungsi untuk preprocessing teks
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()  # Mengubah teks menjadi huruf kecil
        text = re.sub(r'\d+', '', text)  # Menghapus angka
        text = re.sub(r'[^\w\s]', '', text)  # Menghapus tanda baca dan karakter khusus
        text = re.sub(r'\s+', ' ', text).strip()  # Menghapus spasi berlebih

        # Menghapus stopwords
        stopword_factory = StopWordRemoverFactory()
        stopwords = stopword_factory.create_stop_word_remover()
        text = stopwords.remove(text)

        # Stemming
        stemmer_factory = StemmerFactory()
        stemmer = stemmer_factory.create_stemmer()
        text = stemmer.stem(text)

        return text
    return ""

# Fungsi untuk analisis sentimen dengan IndoBERT
def get_sentiment(text):
    if isinstance(text, str) and text.strip():
        clean_text = preprocess_text(text)  # Preprocessing sebelum analisis sentimen
        result = sentiment_pipeline(clean_text)[0]
        return result['label']
    return "Neutral"

# Fungsi untuk membuat WordCloud
def generate_wordcloud(text_data, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(" ".join(text_data))

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=15)
    st.pyplot(plt)

# Tampilan Streamlit
st.title("🔍 Analisis Sentimen Twitter PT KAI")

# Upload file CSV
uploaded_file = st.file_uploader("📂 Upload file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Pastikan hanya kolom "full_text" yang diproses
    if 'full_text' in df.columns:
        st.success("✅ Dataset berhasil dimuat!")

        # Analisis hanya untuk kolom "full_text"
        df = df[['full_text']].copy()
        df['clean_text'] = df['full_text'].apply(preprocess_text)
        df['Sentiment'] = df['clean_text'].apply(get_sentiment)

        # Tampilkan hasil
        st.subheader("📊 Hasil Analisis Sentimen:")
        st.write(df)

        # Download hasil analisis sebagai CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Hasil Sentimen", csv, "hasil_sentimen.csv", "text/csv")

        # Grafik Distribusi Sentimen (Bar Chart)
        st.subheader("📉 Distribusi Sentimen (Grafik Batang)")
        sentiment_counts = df['Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']

        fig_bar = px.bar(sentiment_counts, x='Sentiment', y='Count',
                         color='Sentiment', title="Distribusi Sentimen",
                         text_auto=True, height=500)

        st.plotly_chart(fig_bar, use_container_width=True)

        # Grafik Distribusi Sentimen (Pie Chart)
        st.subheader("🥧 Persentase Sentimen (Pie Chart)")
        fig_pie = px.pie(sentiment_counts, names='Sentiment', values='Count',
                         color='Sentiment', title="Persentase Sentimen")

        st.plotly_chart(fig_pie, use_container_width=True)

        # WordCloud untuk masing-masing sentimen
        st.subheader("🌥️ WordCloud untuk Setiap Sentimen")
        sentiments = ["Positive", "Negative", "Neutral"]

        for sentiment in sentiments:
            words = df[df['Sentiment'] == sentiment]['clean_text']
            if not words.empty:
                generate_wordcloud(words, f"WordCloud untuk Sentimen {sentiment}")

    else:
        st.error("⚠️ Kolom 'full_text' tidak ditemukan dalam dataset. Pastikan file memiliki kolom teks ulasan.")
