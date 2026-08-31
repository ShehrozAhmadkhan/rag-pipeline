from load_data import load_pdf_data,load_text_file_data
from chunking import word_based_chunking
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def batch_based_embedding(chunks,batches = 100):
    all_embeddings = []
    for i in range(0,len(chunks),batches):
        batch = chunks[i:i+batches]
        vectors = client.embeddings.create(model="text-embedding-3-small",input=batch,dimensions=1024)
        for vector in vectors.data:
            all_embeddings.append(vector.embedding)

    return all_embeddings


