import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import string
import requests
from typing import List, Tuple, Optional, Dict, Any

# Stopwords unik
STOPWORDS = set([
    "dan", "atau", "yang", "di", "ke", "dari", "untuk", "dengan", "pada", "adalah", "ini", "itu", "sebagai", "dalam", "oleh", "juga", "karena", "dapat", "akan", "tidak", "bagi"
])

def preprocess(text: str) -> str:
    """Lowercase, hapus tanda baca, dan stopwords."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

def load_data(json_file: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load data bab dan pasal dari file JSON."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    bab_list = data
    pasal_list = []
    for bab in bab_list:
        for pasal in bab['pasal_list']:
            pasal['bab'] = bab['bab']
            pasal_list.append(pasal)
    return bab_list, pasal_list

def build_embeddings(text_list: List[str], model: SentenceTransformer) -> np.ndarray:
    """Bangun embedding dari list teks."""
    preprocessed = [preprocess(t) for t in text_list]
    embeddings = model.encode(preprocessed, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Bangun FAISS index dari embedding."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_bab(query: str, model: SentenceTransformer, bab_index: faiss.IndexFlatL2, bab_list: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], float]:
    """Cari bab paling relevan dengan query."""
    query_vec = model.encode([preprocess(query)])
    D, I = bab_index.search(query_vec, 1)
    return bab_list[I[0][0]], D[0][0]

def search_pasal_in_bab(query: str, model: SentenceTransformer, pasal_list: List[Dict[str, Any]], top_k: int = 3, bab_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Cari pasal relevan, bisa difilter berdasarkan bab."""
    filtered = pasal_list
    if bab_filter:
        filtered = [p for p in pasal_list if bab_filter.lower() == p['bab'].lower()]
        if not filtered:
            filtered = pasal_list
    texts = [p['teks'] for p in filtered]
    preprocessed_texts = [preprocess(t) for t in texts]
    embeddings = model.encode(preprocessed_texts, show_progress_bar=False, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    query_vec = model.encode([preprocess(query)])
    D, I = index.search(query_vec, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        results.append({
            "pasal": filtered[idx]['pasal'],
            "bab": filtered[idx]['bab'],
            "teks": filtered[idx]['teks'],
            "score": dist,
        })
    return results

def generate_answer_ollama(user_question: str, context: str, model_name: str = 'deepseek-r1') -> str:
    """Kirim prompt ke LLM lokal (Ollama) dan ambil jawaban."""
    prompt = (
        f"Berikut adalah beberapa pasal hukum hasil scraping:\n{context}\n\n"
        "Jawablah pertanyaan berikut HANYA berdasarkan pasal-pasal di atas, "
        "gunakan bahasa yang mudah dipahami, dan jangan menambah informasi dari luar konteks.\n"
        f"{user_question}"
    )
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 800
            }
        }
    )
    response.raise_for_status()
    return response.json()["response"].strip()

def chatbot_cli():
    """Main loop CLI chatbot."""
    json_file = "out_dataset_bab_pasal.json"  # sesuaikan dengan nama file JSON hasil ekstraksi Anda
    print("Memuat data dan model embedding ...")
    try:
        bab_list, pasal_list = load_data(json_file)
    except Exception as e:
        print(f"Gagal memuat data: {e}")
        return
    try:
        model = SentenceTransformer('all-mpnet-base-v2')
    except Exception as e:
        print(f"Gagal memuat model embedding: {e}")
        return
    bab_texts = [b['bab'] for b in bab_list]
    bab_embeddings = build_embeddings(bab_texts, model)
    bab_index = build_faiss_index(bab_embeddings)
    print("Chatbot UUD 1945 berbasis Bab dan Pasal siap. Ketik 'exit' untuk keluar.")
    while True:
        query = input("\nTanya sesuatu tentang UUD 1945: ")
        if query.lower() == "exit":
            break
        bab_result, dist = search_bab(query, model, bab_index, bab_list)
        # Threshold untuk memastikan user bertanya dalam konteks bab
        if dist < 1.0:
            print(f"\nTerlihat Anda menanyakan dalam konteks {bab_result['bab']}")
            results = search_pasal_in_bab(query, model, pasal_list, top_k=3, bab_filter=bab_result['bab'])
        else:
            results = search_pasal_in_bab(query, model, pasal_list, top_k=3)
        context = ""
        for res in results:
            # Ambil judul bab dari bab_list
            bab_info = next((b for b in bab_list if b['bab'] == res['bab']), None)
            judul = bab_info['judul'] if bab_info and 'judul' in bab_info else ""
            if judul:
                context += f"Bab: {res['bab']} - {judul} | {res['pasal']}\n{res['teks']}\n"
            else:
                context += f"Bab: {res['bab']} | {res['pasal']}\n{res['teks']}\n"
        try:
            jawaban = generate_answer_ollama(query, context, model_name='deepseek-r1')
            print("\nJawaban:")
            print(jawaban)
        except Exception as e:
            print("\nTerjadi kesalahan saat menghubungi model LLM lokal:", e)
            print("\nBerikut hasil pencarian pasal:")
            for res in results:
                bab_info = next((b for b in bab_list if b['bab'] == res['bab']), None)
                judul = bab_info['judul'] if bab_info and 'judul' in bab_info else ""
                if judul:
                    print(f"Bab: {res['bab']} - {judul} | {res['pasal']}\n{res['teks']}\n(skor kemiripan: {res['score']:.4f})\n")
                else:
                    print(f"Bab: {res['bab']} | {res['pasal']}\n{res['teks']}\n(skor kemiripan: {res['score']:.4f})\n")

def main():
    chatbot_cli()

if __name__ == "__main__":
    main()