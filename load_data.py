import PyPDF2

def load_pdf_data(file_path):
    file = open(file_path,"rb")
    pdfinstring = ""
    pdf = PyPDF2.PdfReader(file)
    for page in pdf.pages:
        text = page.extract_text()
        pdfinstring += text

    return pdfinstring

def load_text_file_data(file_path):
    file = open(file_path,"r",encoding="utf-8")
    textfileinstring = file.read()
    return textfileinstring

