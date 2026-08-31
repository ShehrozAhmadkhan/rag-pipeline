from load_data import load_text_file_data, load_pdf_data

def word_based_chunking(chunksize,overlap,data):
    start = 0
    end = chunksize
    step = chunksize-overlap
    chunks = []
    temp = data.split(" ")
    while start < len(temp):
        chunk = temp[start:end]
        updated_chunk = " ".join(chunk)
        chunks.append(updated_chunk)
        start+=step
        end += step
    return chunks

data = load_pdf_data("medicine.pdf")
chunks = word_based_chunking(100,20,data)
