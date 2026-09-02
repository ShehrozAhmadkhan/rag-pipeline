from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import os
import PyPDF2
import socket
import time

socket.setdefaulttimeout(10)

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pc = Pinecone(api_key = os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-pipeline")

"""
def load_data_from_pdf(file_path):
    file = open(file_path,"rb")
    pdfinstring = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text = page.extract_text()
        pdfinstring += text
    file.close()

    return pdfinstring

def load_data_from_textfile(file_path):
    file = open(file_path,"r",encoding= "utf-8")
    textfileinstring = file.read()
    file.close()
    return textfileinstring


def word_based_chunking(chunk_size,overlap,data):
    start = 0
    end = chunk_size
    step = chunk_size - overlap
    temp = data.split(" ") #[w1,w2,w3,w4,...]
    chunks = []
    while start < len(temp):
        chunk = temp[start:end]
        updated_chunk = " ".join(chunk)
        chunks.append(updated_chunk)
        start += step
        end += step
    return chunks #[c1,c2,c3,...]

def batch_based_embedding(chunks,batches=100):
    all_embeddings = []
    for i in range(0,len(chunks),batches):
        batch = chunks[i:i+batches]
        response = client.embeddings.create(model="text-embedding-3-small",input=batch,dimensions=1024)

        for vector in response.data:
            all_embeddings.append(vector.embedding)

    return all_embeddings

data = load_data_from_textfile("medicine_data.txt")
chunks = word_based_chunking(100,20,data)
embed = batch_based_embedding(chunks)


optimized_vectors = []
for i in range(len(embed)):
    optimized_vectors.append({"id":f"Chunk {i}",
                              "values": embed[i],
                              "metadata":{"text": chunks[i]}})

def batch_based_upserting(index,optimized_vectors,batches=50):
    for i in range(len(optimized_vectors)):
        batch = optimized_vectors[i:i+batches]
        index.upsert(vectors = batch)
    return "DONE"

upsert = batch_based_upserting(index,optimized_vectors)
"""
user_question = input("Ask anything: ")
user_question_embed = client.embeddings.create(model="text-embedding-3-small",input=user_question,dimensions=1024)
user_question_vector = user_question_embed.data[0].embedding

database_query = index.query(vector=user_question_vector,top_k=2,include_metadata=True)
relevant_sentences = []
for i in database_query.matches:
    relevant_sentences.append(i.metadata["text"])

relevant_sentences = "\n".join(relevant_sentences)

prompt = f"""you are an helpfull assistant using the context below answer the question
context: {relevant_sentences}
question: {user_question}
"""

llm = client.chat.completions.create(model="gpt-4o-mini",messages=[{"role":"user","content":prompt}])
answer = llm.choices[0].message.content
print(answer)
