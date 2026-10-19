import pickle

print("Memuat model...")

try:
    # Memuat CountVectorizer dan Model LDA
    with open("count_vectorizer.pkl", "rb") as f:
        cv = pickle.load(f)
    with open("lda_model.pkl", "rb") as f:
        lda = pickle.load(f)

    # Mengambil daftar kata (fitur) dari vectorizer
    feature_names = cv.get_feature_names_out()
    
    print("\n" + "="*40)
    print("HASIL EKSTRAKSI TEMA TOPIK LDA (TOP 10 KATA)")
    print("="*40)

    # Menampilkan 10 kata dengan bobot tertinggi untuk masing-masing topik
    for topic_idx, topic in enumerate(lda.components_):
        # Mengambil indeks 10 kata teratas
        top_features_ind = topic.argsort()[:-11:-1]
        # Mengambil kata aslinya berdasarkan indeks
        top_words = [feature_names[i] for i in top_features_ind]
        
        print(f"Topik {topic_idx + 1}: {', '.join(top_words)}")
        
    print("="*40)

except FileNotFoundError:
    print("Error: File 'count_vectorizer.pkl' atau 'lda_model.pkl' tidak ditemukan.")
    print("Pastikan Anda sudah menjalankan 'python train_model.py' terlebih dahulu.")