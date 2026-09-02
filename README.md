# RAG-Pipeline
Over the past weeks I have been working on building RAG pipeline from scratch.
I have divided my pipeline in different sections, and below is just an overview of how it looks like.

Section 1) importing libraries and loading the API's and connection strings data from .env file into memory and accessing them from memory by using os which is an in-built python module that allows us to access variables from system files.

Section 2) Loading data: I have worked with data in three different formats.
1)Data with-in python script, generated using gpt. 
2) PDF data by using PyPDF2 Library.
3) Text file data.

Section 3) Chunking: I used word based chunking that chunk the data in different chunks, and return it in the form of list. 
[chunk1,chunk2,...]

Section 4) Embedding: I used batch based embedding for embedding my list of chunks into list of vectors and this section returns a list of vectors. [vector1,vector2,...]  
vector1 = [0.3,0.2,0.4,....] #1024 dimensions

Section 5) Upserting the list of vectors in vector database(Pinecone): Before upserting I optimized my list of vectors in the form of id,values, and metadata, and I used batch base upserting to upsert my list of vectors in vector database.

Section 6) User query: Took a user question and embed it into a vector.

Section 7) Database query: Sent a query to vector database to find relevant chunks.

Section 8) LLM integration and User answer generation: Found the relevant chunks, created a prompt and passed the prompt to LLM to generate a clean and well structured user response.
