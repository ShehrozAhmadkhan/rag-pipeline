from load_data import load_pdf_data, load_text_file_data
from chunking import word_based_chunking
from embedding import batch_based_embedding
from dotenv import load_dotenv
from pinecone import Pinecone
import os

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-pipeline")

data = load_pdf_data("medicine.pdf")
data_chunks = word_based_chunking(100, 20, data)
chunks_embed = batch_based_embedding(data_chunks)

optimized_vectors = []
for vector in range(len(chunks_embed)):
    optimized_vectors.append({
        "id": f"chunk_{vector}",
        "values": chunks_embed[vector],
        "metadata": {"text": data_chunks[vector]}
    })

def batch_based_upserting(index, optimized_vectors, batches=100):
    for i in range(0, len(optimized_vectors), batches):
        batch = optimized_vectors[i:i+batches] 
        index.upsert(vectors=batch)
    return "Done"

vector_database = batch_based_upserting(index, optimized_vectors)
print(vector_database)