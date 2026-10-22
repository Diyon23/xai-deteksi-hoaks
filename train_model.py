import pandas as pd
import numpy as np
import re
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

print("Memulai kompilasi dataset...")

files = [
    "antaranews_cleaned_v3.csv", "detik_cleaned_v3.csv", "kompas_cleaned_v3.csv", 
    "tbh_cleaned_v3.csv", "komdigi_hoaks.csv", "dataset_cnn_10k_cleaned.xlsx", 
    "dataset_kompas_4k_cleaned.xlsx", "dataset_tempo_6k_cleaned.xlsx", 
    "dataset_turnbackhoax_10_cleaned.xlsx"
]

dfs = []
for file in files:
    try:
        if file.endswith('.csv'): df = pd.read_csv(file)
        else: df = pd.read_excel(file)
            
        if 'narasi' in df.columns: text_col = 'narasi'
        elif 'FullText' in df.columns: text_col = 'FullText'
        elif 'body_text' in df.columns: text_col = 'body_text'
        else: continue
            
        if 'label' in df.columns: label_col = 'label'
        elif 'hoax' in df.columns: label_col = 'hoax'
        elif 'komdigi' in file.lower(): df['auto_label'] = 1; label_col = 'auto_label'
        else: continue
            
        temp_df = pd.DataFrame({
            'text': df[text_col].astype(str),
            'label': df[label_col].astype(int)
        })
        dfs.append(temp_df)
    except Exception as e:
        pass

final_df = pd.concat(dfs, ignore_index=True).dropna()

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text.lower().strip()

final_df['clean_text'] = final_df['text'].apply(clean_text)

print("Melatih model TF-IDF dan Logistic Regression...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = tfidf_vectorizer.fit_transform(final_df['clean_text'])
y = final_df['label']
clf = LogisticRegression(random_state=42)
clf.fit(X_tfidf, y)

print("Melatih model CountVectorizer dan LDA (2 Tipe)...")
count_vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
X_counts = count_vectorizer.fit_transform(final_df['clean_text'])

# DIUBAH MENJADI 2 KOMPONEN (TIPE)
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X_counts)

slang_dict = {'bgt': 'banget', 'drpd': 'daripada', 'yg': 'yang', 'tdk': 'tidak', 'dgn': 'dengan', 'krn': 'karena'}

with open('tfidf_vectorizer.pkl', 'wb') as f: pickle.dump(tfidf_vectorizer, f)
with open('count_vectorizer.pkl', 'wb') as f: pickle.dump(count_vectorizer, f)
with open('classifier_model.pkl', 'wb') as f: pickle.dump(clf, f)
with open('lda_model.pkl', 'wb') as f: pickle.dump(lda, f)
with open('slang_dict.pkl', 'wb') as f: pickle.dump(slang_dict, f)

print("✅ Selesai! Model 2 Tipe berhasil diekspor.")