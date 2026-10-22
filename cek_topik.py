import pickle
import pandas as pd
import numpy as np

print("Memuat model dan dataset...")

try:
    # 1. Memuat CountVectorizer, Model LDA, dan Dataset pelatihan
    with open("count_vectorizer.pkl", "rb") as f:
        cv = pickle.load(f)
    with open("lda_model.pkl", "rb") as f:
        lda = pickle.load(f)
        
    # Memuat salah satu file dataset untuk mengambil contoh teks
    df = pd.read_csv("antaranews_cleaned_v3.csv")
    if 'narasi' in df.columns:
        texts = df['narasi'].astype(str).tolist()
    elif 'body_text' in df.columns:
        texts = df['body_text'].astype(str).tolist()
    else:
        texts = df.iloc[:, 0].astype(str).tolist()

    # Transformasi dokumen ke ruang topik LDA
    X_counts = cv.transform(texts)
    doc_topics = lda.transform(X_counts)

    feature_names = cv.get_feature_names_out()
    
    print("\n" + "="*70)
    print("ANALISIS ISI DAN KARAKTERISTIK 2 TIPE LDA")
    print("="*70)

    # 2. Iterasi untuk masing-masing tipe (Tipe 1 & Tipe 2)
    for topic_idx, topic in enumerate(lda.components_):
        # Ambil 10 kata kunci teratas
        top_features_ind = topic.argsort()[:-11:-1]
        top_words = [feature_names[i] for i in top_features_ind]
        
        # Penamaan Tipe sesuai konteks hoaks/fakta
        if topic_idx == 0:
            tipe_nama = "Tipe 1: Penyebab Hoaks (Klaim Palsu / Sensasional)"
        else:
            tipe_nama = "Tipe 2: Berita Fakta & Informasi Kredibel"

        print(f"\n🔹 {tipe_nama}")
        print(f"Top 10 Kata Kunci: {', '.join(top_words)}")
        
        # Cari dokumen/berita yang paling dominan di tipe ini
        dominant_doc_idx = np.argmax(doc_topics[:, topic_idx])
        sample_text = texts[dominant_doc_idx]
        
        # Batasi panjang teks untuk preview
        if len(sample_text) > 250:
            sample_text = sample_text[:250] + "..."
            
        print(f"Contoh Isi Berita Representatif:\n\"{sample_text}\"")
        print("-" * 70)

except FileNotFoundError as e:
    print(f"Error: {e}. Pastikan file model dan dataset tersedia di folder proyek.")
except Exception as e:
    print(f"Terjadi kesalahan: {e}")