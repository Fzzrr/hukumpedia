import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    bab_list = data
    pasal_list = []
    for bab in bab_list:
        for pasal in bab['pasal_list']:
            # Tambahkan info bab ke setiap pasal
            pasal['bab'] = bab['bab']
            pasal_list.append(pasal)
    return bab_list, pasal_list

def build_embeddings(text_list, model):
    embeddings = model.encode(text_list, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_bab(query, model, bab_index, bab_list):
    query_vec = model.encode([query])
    D, I = bab_index.search(query_vec, 1)
    return bab_list[I[0][0]], D[0][0]

def search_pasal_in_bab(query, model, pasal_list, top_k=3, bab_filter=None):
    # Filter pasal berdasarkan bab jika bab_filter diberikan
    if bab_filter:
        filtered = [p for p in pasal_list if bab_filter.lower() == p['bab'].lower()]
        if not filtered:
            filtered = pasal_list
    else:
        filtered = pasal_list

    texts = [p['teks'] for p in filtered]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    query_vec = model.encode([query])
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

def main():
    json_file = "out_dataset_bab_pasal.json"  # sesuaikan dengan nama file JSON hasil ekstraksi Anda

    print("Memuat data dan model embedding ...")
    bab_list, pasal_list = load_data(json_file)

    model = SentenceTransformer('all-mpnet-base-v2')

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

        print("\nHasil pencarian:")
        for res in results:
            print(f"Bab: {res['bab']} | {res['pasal']}\n{res['teks']}\n(skor kemiripan: {res['score']:.4f})\n")

if __name__ == "__main__":
    main()
