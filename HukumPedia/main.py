from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever  # Import retriever dari vector.py yang sudah berisi vector store dan indexing dokumen

# Inisialisasi model LLM Ollama dengan model "llama3.2"
model = OllamaLLM(model="deepseek-r1")

# Template prompt untuk memberikan konteks dokumen hukum dan pertanyaan user
template = """
You are an expert in Indonesian legal documents and government regulations.

Here are some relevant legal document excerpts: {reviews}

Please answer the following legal question accurately using the above information:
{question}
"""

# Buat prompt dari template
prompt = ChatPromptTemplate.from_template(template)

def main():
    while True:
        print("\n\n-------------------------------")
        question = input("Tanyakan pertanyaan hukum Anda (ketik 'q' untuk keluar): ")
        if question.lower() == "q":
            break

        # Ambil dokumen terkait dari retriever berbasis Chroma
        docs = retriever.get_relevant_documents(question)

        # Gabungkan isi dokumen untuk menjadi konteks pada prompt
        reviews = "\n\n".join([doc.page_content for doc in docs])

        # Format prompt dengan konteks isi dokumen dan pertanyaan
        formatted_prompt = prompt.format_prompt(reviews=reviews, question=question)

        # Dapatkan jawaban dari model dengan mengirim pesan prompt
        response = model.invoke(formatted_prompt.to_messages())

        print("\nJawaban:\n", response)


if __name__ == "__main__":
    main()
