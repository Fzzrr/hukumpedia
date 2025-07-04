from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import os
import pandas as pd

# Load ketiga file CSV dan gabungkan
df_train = pd.read_csv("datasets/train.csv")
df_test = pd.read_csv("datasets/test.csv")
df_val = pd.read_csv("datasets/val.csv")

df = pd.concat([df_train, df_test, df_val], ignore_index=True)

# Inisialisasi embeddings dengan model Ollama
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Lokasi folder database vector store yang persistent
db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

def batch_documents(documents, ids, batch_size=5000):
    """Generator untuk memecah list documents dan ids menjadi batch-batch kecil"""
    for i in range(0, len(documents), batch_size):
        yield documents[i:i+batch_size], ids[i:i+batch_size]

# Inisialisasi vector store Chroma dengan embeddings dan direktori persistensi
vector_store = Chroma(
    collection_name="hukum_pedia",
    persist_directory=db_location,
    embedding_function=embeddings,
)

if add_documents:
    documents = []
    ids = []

    # Pastikan kolom 'concatenated' ada dan berisi teks dokumen dari dataframe
    for i, row in df.iterrows():
        document = Document(
            page_content=row["concatenated"],  # Ganti dengan nama kolom sesuai data csv Anda
            metadata={},  # bisa diisi metadata jika perlu
            id=str(i)
        )
        documents.append(document)
        ids.append(str(i))

    # Kirim batch kecil ke vector store
    batch_size = 5000  # ukuran batch < 5461 (batas maksimal dari ChromaDB)
    for docs_batch, ids_batch in batch_documents(documents, ids, batch_size):
        vector_store.add_documents(documents=docs_batch, ids=ids_batch)
    
    # Simpan indeks ke disk setelah semua batch selesai ditambahkan
    vector_store.persist()

# Buat retriever untuk pencarian dokumen
retriever = vector_store.as_retriever(search_kwargs={"k": 15})
