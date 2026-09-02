import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-pipeline")


user_question = input("Enter your question: ")
user_question_embed = client.embeddings.create(model="text-embedding-3-small",input=user_question,dimensions=1024)
user_question_vector = user_question_embed.data[0].embedding

database_query = index.query(vector=user_question_vector,top_k=2,include_metadata=True)
relevant_sentence = []
for i in database_query.matches:
    relevant_sentence.append(i.metadata["text"])

relevant_chunks = "\n".join(relevant_sentence)

prompt = """ you are an helpfull assistant using the context below answer the question
context : f"{relevant_chunks}"
user question: f"{user_question}"
"""

llm = client.chat.completions.create(model="gpt-4o-mini",messages=[{"role":"user", "content":prompt}])
answer = llm.choices[0].message.content
print(answer)
