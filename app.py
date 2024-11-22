import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

# Load Data
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Fungsi Narasi Otomatis
def generate_insights(data, sentiment_filter, model_filter):
    insights = []
    if model_filter != "Semua":
        insights.append(f"Model handphone yang dipilih adalah **{model_filter}**.")
    if sentiment_filter != "Semua":
        count = len(data[data["PredictedSentiment"] == sentiment_filter])
        insights.append(f"Jumlah komentar dengan sentimen **{sentiment_filter}** adalah **{count}**.")
    else:
        total = len(data)
        insights.append(f"Jumlah total komentar yang dianalisis adalah **{total}**.")
    if data.empty:
        insights.append("Data kosong untuk filter yang dipilih.")
    return insights

# Fungsi Visualisasi
def plot_sentiment_distribution(data):
    if data.empty:
        st.warning("Data kosong untuk filter ini.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=data, x="PredictedSentiment", order=data["PredictedSentiment"].value_counts().index, palette="viridis", ax=ax)
    ax.set_title("Distribusi Sentimen", fontsize=14)
    ax.set_xlabel("Sentimen", fontsize=12)
    ax.set_ylabel("Jumlah Komentar", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

def generate_wordcloud(data):
    if data.empty:
        st.warning("Data kosong untuk filter ini.")
        return
    text = " ".join(data["cleaned_text_2"].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title("Word Cloud", fontsize=14)
    ax.axis("off")
    st.pyplot(fig)

def plot_top_words(data, sentiment, model=None, top_n=10):
    if data.empty:
        st.warning("Data kosong untuk filter ini.")
        return
    text = " ".join(data["cleaned_text_2"].dropna())
    word_freq = pd.Series(text.split()).value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=word_freq.values, y=word_freq.index, palette="viridis", ax=ax)
    title = f"Top {top_n} Words untuk Sentimen: {sentiment.capitalize()}"
    if model:
        title += f" (Model: {model})"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Frekuensi", fontsize=12)
    ax.set_ylabel("Kata", fontsize=12)
    st.pyplot(fig)

def plot_sentiment_correlation(data):
    if data.empty:
        st.warning("Data kosong untuk filter ini.")
        return
    pivot_table = data.pivot_table(index="phoneModel", columns="PredictedSentiment", aggfunc="size", fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt="d", cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Korelasi Sentimen Antar Model", fontsize=14)
    st.pyplot(fig)

def plot_comment_length_distribution(data):
    if data.empty:
        st.warning("Data kosong untuk filter ini.")
        return
    data["comment_length"] = data["cleaned_text_2"].apply(lambda x: len(str(x).split()))
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=data, x="PredictedSentiment", y="comment_length", palette="coolwarm", ax=ax)
    ax.set_title("Distribusi Panjang Komentar Berdasarkan Sentimen", fontsize=14)
    ax.set_xlabel("Sentimen", fontsize=12)
    ax.set_ylabel("Panjang Komentar (Jumlah Kata)", fontsize=12)
    st.pyplot(fig)

# Radar Chart: Visualisasi distribusi sentimen antar model
def plot_sentiment_radar_chart(data):
    if data.empty:
        st.warning("Data kosong untuk filter ini.")
        return
    models = data['phoneModel'].unique()
    sentiments = data['PredictedSentiment'].unique()

    # Prepare the data for radar chart
    radar_data = pd.DataFrame(index=models, columns=sentiments, dtype=int)
    for model in models:
        for sentiment in sentiments:
            radar_data.loc[model, sentiment] = len(data[(data['phoneModel'] == model) & (data['PredictedSentiment'] == sentiment)])

    # Plot Radar Chart
    categories = list(radar_data.columns)
    values = radar_data.loc[models[0], :].values.tolist()
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_title(f"Radar Chart untuk Model {models[0]}", fontsize=14)
    st.pyplot(fig)

# Fungsi Narasi Otomatis
def generate_insights(data, sentiment_filter, model_filter, compare_data=None, compare_model_filter=None):
    insights = []

    # Narasi untuk model yang dipilih
    if model_filter == "Semua":
        insights.append("Semua model handphone dipilih.")
        if sentiment_filter == "Semua":
            total = len(data)
            insights.append(f"Jumlah total komentar yang dianalisis untuk semua model adalah **{total}**.")
        else:
            count = len(data[data["PredictedSentiment"] == sentiment_filter])
            insights.append(f"Jumlah komentar dengan sentimen **{sentiment_filter}** untuk semua model adalah **{count}**.")
    else:
        insights.append(f"Model handphone yang dipilih adalah **{model_filter}**.")
        if sentiment_filter != "Semua":
            count = len(data[data["PredictedSentiment"] == sentiment_filter])
            insights.append(f"Jumlah komentar dengan sentimen **{sentiment_filter}** untuk {model_filter} adalah **{count}**.")
        else:
            total = len(data)
            insights.append(f"Jumlah total komentar yang dianalisis untuk {model_filter} adalah **{total}**.")
    
    # Narasi jika data kosong untuk filter yang dipilih
    if data.empty:
        insights.append("Data kosong untuk filter yang dipilih.")
    
    # Jika ada perbandingan, tambahkan narasi untuk model perbandingan
    if compare_data is not None and compare_model_filter:
        if compare_model_filter == "Semua":
            insights.append("Perbandingan dilakukan dengan semua model handphone.")
            if sentiment_filter == "Semua":
                total_compare = len(compare_data)
                insights.append(f"Jumlah total komentar yang dianalisis untuk semua model adalah **{total_compare}**.")
            else:
                count_compare = len(compare_data[compare_data["PredictedSentiment"] == sentiment_filter])
                insights.append(f"Jumlah komentar dengan sentimen **{sentiment_filter}** untuk semua model adalah **{count_compare}**.")
        else:
            insights.append(f"Perbandingan dengan model handphone **{compare_model_filter}**.")
            if sentiment_filter != "Semua":
                count_compare = len(compare_data[compare_data["PredictedSentiment"] == sentiment_filter])
                insights.append(f"Jumlah komentar dengan sentimen **{sentiment_filter}** untuk {compare_model_filter} adalah **{count_compare}**.")
            else:
                total_compare = len(compare_data)
                insights.append(f"Jumlah total komentar yang dianalisis untuk {compare_model_filter} adalah **{total_compare}**.")
        
        # Narasi untuk data kosong pada model perbandingan
        if compare_data.empty:
            insights.append(f"Data kosong untuk filter pada model {compare_model_filter}.")
    
    return insights

# Fungsi Pencarian Komentar dengan Filter Model
def search_comments_with_filter(data, keyword, model_filter):
    # Filter data berdasarkan model
    if model_filter != "Semua":
        filtered_data = data[data["phoneModel"] == model_filter]
    else:
        filtered_data = data

    # Filter berdasarkan kata kunci pada cleaned_text_2
    return filtered_data[filtered_data["cleaned_text_2"].str.contains(keyword, case=False, na=False)]

# Aplikasi Streamlit
# Tema Streamlit
st.set_page_config(page_title="Analisis Sentimen YouTube", layout="wide", initial_sidebar_state="expanded")

# Halaman Utama
st.title("📊 Analisis Sentimen Komentar YouTube")
st.markdown(
    """
    Selamat datang di aplikasi analisis sentimen komentar YouTube! 
    Jelajahi berbagai model handphone dengan sentimen komentar pengguna, 
    distribusi kata, dan korelasi antar model.
    """
)

st.sidebar.title("Aplikasi Analisis Sentimen")  # Menambahkan judul aplikasi di atas
st.sidebar.header("🔍 Filter")
st.sidebar.info("Gunakan filter di bawah untuk menyaring data komentar berdasarkan model, sentimen, dan kata kunci.")
# Tentukan path file CSV yang sudah ada
data_file_path = "df_predicted_8-2_OK.csv"  # Ganti dengan path file CSV yang sesuai

# Memuat data langsung dari file tanpa memerlukan upload dari pengguna
data = load_data(data_file_path)

# Pencarian Komentar
search_keyword = st.sidebar.text_input("Cari Komentar Berdasarkan Kata Kunci")

# Pastikan file data sudah dimuat
if not data.empty:
    # Validasi Kolom Data
    expected_columns = {"phoneModel", "PredictedSentiment", "cleaned_text_2"}
    if not expected_columns.issubset(set(data.columns)):
        st.error(f"File tidak valid. Pastikan memiliki kolom: {', '.join(expected_columns)}")
    else:
        # Pilih Model Handphone
        all_models = data["phoneModel"].unique()
        model_filter = st.sidebar.selectbox("Pilih Model Handphone", ["Semua"] + list(all_models))
        
        # Lakukan pencarian jika ada kata kunci
        if search_keyword:
            search_results = search_comments_with_filter(data, search_keyword, model_filter)
            st.subheader(f"Komentar untuk smartphone {model_filter} yang mengandung kata kunci: '{search_keyword}'")
            st.write(search_results[["phoneModel", "PredictedSentiment", "cleaned_text_2"]])

        st.sidebar.subheader("Opsi Filter")
        sentiment_filter = st.sidebar.selectbox("Pilih Sentimen", ["Semua"] + list(data["PredictedSentiment"].unique()))
        
        compare_option = st.sidebar.checkbox("Bandingkan dengan model lain?")
        model_filter_compare = None
        if compare_option:
            model_filter_compare = st.sidebar.selectbox("Pilih Model Handphone untuk Perbandingan", list(all_models))

        # Filter Data Berdasarkan Pilihan
        filtered_data = data.copy()
        if model_filter != "Semua":
            filtered_data = filtered_data[filtered_data["phoneModel"] == model_filter]
        if sentiment_filter != "Semua":
            filtered_data = filtered_data[filtered_data["PredictedSentiment"] == sentiment_filter]

        filtered_data_compare = None
        if model_filter_compare:
            filtered_data_compare = data[data["phoneModel"] == model_filter_compare]

        # Generate narasi berdasarkan filter
        insights = generate_insights(filtered_data, sentiment_filter, model_filter, filtered_data_compare, model_filter_compare)
        
        # Tampilkan narasi
        if insights:
            for insight in insights:
                st.markdown(f"- {insight}")
        else:
            st.warning("Tidak ada narasi yang dapat dihasilkan untuk filter ini.")

        # Tabs untuk Organisasi
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📈 Distribusi Sentimen", 
            "☁️ Word Cloud", 
            "🔥 Korelasi Sentimen", 
            "🔤 Top Words", 
            "📊 Panjang Komentar", 
            "🛸 Radar Chart",
            "🎲 Komentar Acak"
        ])

        with tab1:
            st.subheader("Distribusi Sentimen")
            plot_sentiment_distribution(filtered_data)
            if compare_option and filtered_data_compare is not None:
                st.subheader(f"Distribusi Sentimen - Perbandingan dengan {model_filter_compare}")
                plot_sentiment_distribution(filtered_data_compare)

        with tab2:
            st.subheader("Word Cloud")
            generate_wordcloud(filtered_data)
            if compare_option and filtered_data_compare is not None:
                st.subheader(f"Word Cloud - Perbandingan dengan {model_filter_compare}")
                generate_wordcloud(filtered_data_compare)

        with tab3:
            st.subheader("Korelasi Sentimen Antar Model")
            plot_sentiment_correlation(filtered_data)
            if compare_option and filtered_data_compare is not None:
                st.subheader(f"Korelasi Sentimen - Perbandingan dengan {model_filter_compare}")
                plot_sentiment_correlation(filtered_data_compare)

        with tab4:
            st.subheader("Top Words Berdasarkan Sentimen dan Model")
            if sentiment_filter == "Semua":
                st.info("Silakan pilih sentimen terlebih dahulu di sidebar.")
            else:
                plot_top_words(filtered_data, sentiment=sentiment_filter, model=model_filter if model_filter != "Semua" else None)
                if compare_option and filtered_data_compare is not None:
                    plot_top_words(filtered_data_compare, sentiment=sentiment_filter, model=model_filter_compare)

        with tab5:
            st.subheader("Distribusi Panjang Komentar Berdasarkan Sentimen")
            plot_comment_length_distribution(filtered_data)
            if compare_option and filtered_data_compare is not None:
                st.subheader(f"Distribusi Panjang Komentar - Perbxandingan dengan {model_filter_compare}")
                plot_comment_length_distribution(filtered_data_compare)

        with tab6:
            st.subheader("Radar Chart Sentimen")
            plot_sentiment_radar_chart(filtered_data)
            if compare_option and filtered_data_compare is not None:
                st.subheader(f"Radar Chart - Perbandingan dengan {model_filter_compare}")
                plot_sentiment_radar_chart(filtered_data_compare)
        
        with tab7:
            st.subheader("Komentar Acak")
        if filtered_data.empty:
            st.warning("Tidak ada komentar yang tersedia untuk filter yang dipilih.")
        else:
            # Ambil 20 komentar acak dari data yang sudah difilter
            random_comments = filtered_data.sample(n=100)
            st.write(random_comments[["phoneModel", "PredictedSentiment", "cleaned_text_2"]])

            if compare_option and filtered_data_compare is not None:
                st.subheader(f"Komentar Acak - Perbandingan dengan {model_filter_compare}")
                random_comments_compare = filtered_data_compare.sample(n=100)
                st.write(random_comments_compare[["phoneModel", "PredictedSentiment", "cleaned_text_2"]])
