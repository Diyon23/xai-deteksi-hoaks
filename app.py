import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import streamlit.components.v1 as components
import lime.lime_text
import shap
import eli5
from eli5.lime import TextExplainer

# ==========================================
# 1. LOAD MODEL
# ==========================================
@st.cache_resource
def load_models():
    try:
        with open("tfidf_vectorizer.pkl", "rb") as f: tfidf_vectorizer = pickle.load(f)
        with open("count_vectorizer.pkl", "rb") as f: count_vectorizer = pickle.load(f)
        with open("classifier_model.pkl", "rb") as f: classifier_model = pickle.load(f)
        with open("lda_model.pkl", "rb") as f: lda_model = pickle.load(f)
        with open("slang_dict.pkl", "rb") as f: slang_dict = pickle.load(f)
    except FileNotFoundError:
        st.error("File model .pkl belum ditemukan! Silakan jalankan 'python train_model.py' terlebih dahulu di terminal.")
        tfidf_vectorizer, count_vectorizer, classifier_model, lda_model, slang_dict = None, None, None, None, {}
    return tfidf_vectorizer, count_vectorizer, classifier_model, lda_model, slang_dict

tfidf_vectorizer, count_vectorizer, classifier_model, lda_model, slang_dict = load_models()

def predict_probability_wrapper(texts):
    vec_texts = tfidf_vectorizer.transform(texts)
    return classifier_model.predict_proba(vec_texts)

# ==========================================
# 2. PENGATURAN HALAMAN & KUSTOMISASI CSS
# ==========================================
st.set_page_config(page_title="Sistem Deteksi Hoaks & Audit XAI", layout="wide")

st.markdown("""
<style>
.main-header { background-color: #1E293B; color: white; padding: 15px 20px; border-radius: 8px; margin-bottom: 25px; }
.sub-header-text { font-size: 14px; font-weight: normal; color: #94a3b8; }
.status-box-red { border: 2px solid #ff4b4b; background-color: #fffafa; padding: 15px; border-radius: 8px; text-align: center; color: #cc0000; }
.status-box-blue { border: 2px solid #1f77b4; background-color: #f0f7ff; padding: 15px; border-radius: 8px; text-align: center; color: #0055a4; }
.status-title { font-size: 14px; font-weight: bold; margin-bottom: 5px; color: #333; }
.status-value { font-size: 22px; font-weight: bold; }
.eli5-box-yellow { border: 1px solid #f4c430; background-color: #fffdf5; padding: 20px; border-radius: 8px; margin-top: 15px; }
.table-header { padding: 12px; border-radius: 6px; font-weight: bold; text-align: center; margin-bottom: 10px; }
.header-hoaks { background-color: #ffeaea; color: #d62728; }
.header-fakta { background-color: #eaffea; color: #2ca02c; }
.header-overlap { background-color: #fff4cc; color: #b8860b; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h2 style="margin:0;">SISTEM DETEKSI HOAKS & AUDIT TRANSPARANSI XAI</h2>
    <div class="sub-header-text">Program Studi Informatika - Universitas Kristen Petra | Multi-Method: LIME, SHAP, ELI5 & LDA</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 3. TATA LETAK UTAMA (KIRI & KANAN)
# ==========================================
col_left, col_right = st.columns([1.2, 2.8])

with col_left:
    st.markdown("#### Semakin Banyak Jumlah Kata Pada Narasi Berita Yang Anda Masukan maka semakin Presisi Hasilnya")
    raw_text = st.text_area("Masukkan Narasi / Artikel Berita:", height=200, placeholder='"Masukkan teks berita di sini..."')
    xai_method = st.radio("Pilih Metode Interpretasi (XAI) disarankan menggunakan mode komparasi semua karena akan mengeluarkan semua metodenya secara sekaligus di masing-masing tab :", options=["LIME (Local Surrogate)", "SHAP (Shapley Values)", "ELI5 (Text Highlighting)", "Mode Komparasi Semua"])
    top_k = st.slider("Jumlah Fitur Kata Kunci (Top-K):", min_value=5, max_value=20, value=10)
    analyze_button = st.button("🔍 ANALISIS TRANSPARANSI", use_container_width=True, type="primary")

with col_right:
    status_color_class, status_label, status_val, topic_val = "status-box-blue", "SIAP DIANALISIS", "Menunggu Teks...", "Menunggu Proses..."
    
    if analyze_button and raw_text and tfidf_vectorizer:
        clean_text = re.sub(r'[^a-zA-Z\s]', ' ', raw_text).lower()
        normalized_tokens = [slang_dict.get(word, word) for word in clean_text.split()]
        final_text = " ".join(normalized_tokens)
        
        # Prediksi Klasifikasi menggunakan TF-IDF
        vec_text = tfidf_vectorizer.transform([final_text])
        pred_prob = classifier_model.predict_proba(vec_text)[0]
        hoax_prob = pred_prob[1]
        
        if hoax_prob >= 0.5:
            status_color_class, status_label, status_val = "status-box-red", "STATUS PREDIKSI KLASIFIKASI:", f"🚨 Berita ini {hoax_prob*100:.1f}% HOAKS"
        else:
            status_color_class, status_label, status_val = "status-box-blue", "STATUS PREDIKSI KLASIFIKASI:", f"✅ Berita ini {(1-hoax_prob)*100:.1f}% FAKTA"
            
        # Prediksi Aspek LDA (2 Tipe)
        vec_text_lda = count_vectorizer.transform([final_text])
        lda_probs = lda_model.transform(vec_text_lda)[0]
        topic_id = np.argmax(lda_probs) + 1
        
        # Penamaan Tipe berdasarkan karakteristik hoaks dan fakta
        if topic_id == 1:
            topic_val = "🏷️ Tipe 1: Penyebab Hoaks (Klaim Palsu / Sensasional)"
        else:
            topic_val = "🏷️ Tipe 2: Informasi Fakta & Berita Kredibel"
    
    top_col1, top_col2 = st.columns(2)
    with top_col1:
        st.markdown(f'<div class="{status_color_class}"><div class="status-title">{status_label}</div><div class="status-value">{status_val}</div></div>', unsafe_allow_html=True)
    with top_col2:
        st.markdown(f'<div class="status-box-blue"><div class="status-title">ASPEK TEMATIK BERITA (LDA):</div><div class="status-value">{topic_val}</div></div>', unsafe_allow_html=True)
        
    st.write("") 
    
    tab1, tab2, tab3, tab4 = st.tabs(["Tab 1: LIME", "Tab 2: SHAP", "Tab 3: ELI5", "Tab 4: Komparasi"])
    
    with tab1:
        st.markdown("**Visualisasi Pengaruh Kata terhadap Prediksi (Local Feature Importance):**")
        if analyze_button and raw_text and tfidf_vectorizer and ("LIME" in xai_method or "Semua" in xai_method):
            with st.spinner("Memproses LIME..."):
                explainer = lime.lime_text.LimeTextExplainer(class_names=['Fakta', 'Hoaks'])
                lime_exp = explainer.explain_instance(final_text, predict_probability_wrapper, num_features=top_k)
                components.html(lime_exp.as_html(), height=350, scrolling=True)
        else:
            st.info("Visualisasi LIME akan ditampilkan di sini.")

    with tab2:
        st.markdown("**Nilai Marginal Shapley Values (SHAP):**")
        if analyze_button and raw_text and tfidf_vectorizer and ("SHAP" in xai_method or "Semua" in xai_method):
            with st.spinner("Memproses SHAP..."):
                masker = shap.maskers.Text(r"\s+")
                def f(x):
                    return classifier_model.predict_proba(tfidf_vectorizer.transform(x))
                
                explainer = shap.Explainer(f, masker, output_names=['Fakta', 'Hoaks'])
                shap_values = explainer([final_text])
                
                shap_html = f"<head>{shap.getjs()}</head><body>{shap.plots.text(shap_values[0], display=False)}</body>"
                components.html(shap_html, height=350, scrolling=True)
        else:
            st.info("Visualisasi SHAP akan ditampilkan di sini.")

    with tab3:
        if analyze_button and raw_text and tfidf_vectorizer and ("ELI5" in xai_method or "Semua" in xai_method):
            with st.spinner("Memproses ELI5..."):
                try:
                    # Menggunakan metode TextExplainer sesuai pendekatan Post-Hoc Black-Box Skripsi
                    te = TextExplainer(random_state=42)
                    te.fit(final_text, predict_probability_wrapper)
                    
                    eli5_exp = te.explain_prediction(target_names=['Fakta', 'Hoaks'], top=top_k)
                    html_data = eli5.format_as_html(eli5_exp)
                    
                    st.markdown('<div class="eli5-box-yellow"><b>Inspeksi Leksikal Dokumen Berita (ELI5 Text Highlighting):</b></div>', unsafe_allow_html=True)
                    components.html(html_data, height=400, scrolling=True)
                except Exception as e:
                    st.error(f"Gagal memuat ELI5: {e}")
        else:
             st.markdown("<div class='eli5-box-yellow'><b>Inspeksi Leksikal Dokumen Berita:</b><br><br><span style='color: gray;'>Menunggu masukan teks...</span></div>", unsafe_allow_html=True)
                
    with tab4:
        st.markdown("**Hasil Komparasi Metode XAI (Evaluasi Kuantitatif):**")
        if analyze_button and raw_text and tfidf_vectorizer and ("Semua" in xai_method):
            with st.spinner("Menghitung Jaccard Similarity Index..."):
                try:
                    def clean_token(token):
                        if isinstance(token, bytes):
                            token = token.decode('utf-8', errors='ignore')
                        else:
                            token = str(token)
                        return re.sub(r'[^a-zA-Z]', '', token).lower()

                    # Ekstraksi Top-K Kata (LIME vs SHAP)
                    lime_words = [x[0] for x in lime_exp.as_list()[:top_k]]
                    lime_set = set([clean_token(w) for w in lime_words if clean_token(w)])
                    
                    shap_tokens = np.array(shap_values[0].data)
                    shap_weights = np.array(shap_values[0].values)
                    
                    valid_indices = [i for i, token in enumerate(shap_tokens) if clean_token(token)]
                    valid_tokens = shap_tokens[valid_indices]
                    valid_weights = shap_weights[valid_indices]
                    
                    top_shap_indices = np.argsort(np.abs(valid_weights))[::-1][:top_k]
                    shap_words = [valid_tokens[i] for i in top_shap_indices]
                    shap_set = set([clean_token(w) for w in shap_words if clean_token(w)])
                    
                    # Hitung Jaccard Similarity Index (Persamaan 3.4)
                    intersection = lime_set.intersection(shap_set)
                    union = lime_set.union(shap_set)
                    jaccard_index = len(intersection) / len(union) if len(union) > 0 else 0
                    
                    st.success(f"**Jaccard Similarity Index (LIME vs SHAP): {jaccard_index:.2f}**")
                    st.progress(jaccard_index)
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.markdown("🟢 **Top Kata LIME:**")
                        for w in lime_set: st.markdown(f"- {w}")
                    with col_b:
                        st.markdown("🔵 **Top Kata SHAP:**")
                        for w in shap_set: st.markdown(f"- {w}")
                    with col_c:
                        st.markdown("🟡 **Irisan (Kesepakatan):**")
                        if intersection:
                            for w in intersection: st.markdown(f"- **{w}**")
                        else:
                            st.markdown("- *Tidak ada kesepakatan kata*")
                except Exception as e:
                    st.error(f"Gagal memproses metrik komparasi: {e}")
        else:
             st.info("Pilih opsi 'Mode Komparasi Semua' lalu klik 'Analisis Transparansi' untuk melihat tingkat kesepakatan fitur (Jaccard Index).")

# ==========================================
# 4. TABEL AUDIT LINGUISTIK TOPIK
# ==========================================
st.markdown("---")
st.markdown("#### TABEL AUDIT LINGUISTIK TOPIK: PEMETAAN FITUR KATA KUNCI")

hoax_words, fakta_words, overlap_words = [], [], []

if analyze_button and raw_text and tfidf_vectorizer:
    try:
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Ekstraksi Bobot Koefisien Model (Mendukung Naive Bayes & Logistic Regression)
        if hasattr(classifier_model, "coef_"):
            coefs = classifier_model.coef_[0]
        elif hasattr(classifier_model, "feature_log_prob_"):
            coefs = classifier_model.feature_log_prob_[1] - classifier_model.feature_log_prob_[0]
        else:
            coefs = np.zeros(len(feature_names))

        vec_array = vec_text.toarray()[0]
        active_indices = [i for i, val in enumerate(vec_array) if val > 0]
        active_word_weights = [(feature_names[i], coefs[i]) for i in active_indices]
        
        if active_word_weights:
            # 1. Kata Spesifik Hoaks (Bobot > 0, Diambil Top-K)
            sorted_hoax = sorted([item for item in active_word_weights if item[1] > 0], key=lambda x: x[1], reverse=True)
            hoax_words = [w for w, _ in sorted_hoax[:top_k]]
            
            # 2. Kata Spesifik Fakta (Bobot < 0, Diambil Top-K)
            sorted_fakta = sorted([item for item in active_word_weights if item[1] < 0], key=lambda x: x[1])
            fakta_words = [w for w, _ in sorted_fakta[:top_k]]
            
            # 3. Kata Tumpang-Tindih / Netral (Sisanya, bobot mendekati 0)
            used_words = set(hoax_words + fakta_words)
            remaining_words = [item for item in active_word_weights if item[0] not in used_words]
            sorted_overlap = sorted(remaining_words, key=lambda x: abs(x[1]))
            overlap_words = [w for w, _ in sorted_overlap[:top_k]]
            
    except Exception as e:
        st.error(f"Error ekstraksi audit: {e}")

col_hoaks, col_fakta, col_overlap = st.columns(3)

with col_hoaks:
    st.markdown('<div class="table-header header-hoaks">🔴 Kata Spesifik Hoaks (Top Weight)</div>', unsafe_allow_html=True)
    if analyze_button and raw_text:
        if hoax_words:
            for w in hoax_words: st.markdown(f"- **{w}**")
        else:
            st.markdown("- *Tidak ada kata spesifik hoaks (bobot positif) pada teks ini*")
    else:
        st.markdown("- mujarab (+0.42)\n- semalam (+0.31)")

with col_fakta:
    st.markdown('<div class="table-header header-fakta">🟢 Kata Spesifik Non-Hoaks / Fakta</div>', unsafe_allow_html=True)
    if analyze_button and raw_text:
        if fakta_words:
            for w in fakta_words: st.markdown(f"- **{w}**")
        else:
            st.markdown("- *Tidak ada kata spesifik fakta (bobot negatif) pada teks ini*")
    else:
        st.markdown("- uji klinis (-0.38)\n- bpom (-0.29)")

with col_overlap:
    st.markdown('<div class="table-header header-overlap">🟡 Kata Tumpang-Tindih (Overlapping)</div>', unsafe_allow_html=True)
    if analyze_button and raw_text:
        if overlap_words:
            for w in overlap_words: st.markdown(f"- **{w}** (Netral/Topik)")
        else:
            st.markdown("- *Tidak ada kata tumpang tindih tersisa*")
    else:
        st.markdown("- kementerian (Kontekstual)\n- obat (Netral)")