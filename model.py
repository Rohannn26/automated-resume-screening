#pip3 install PyPDF2

import PyPDF2

def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text()
    return text

extracted_text = extract_text_from_pdf('tyler.pdf')
#print(extracted_text)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_text_from_file(filename):
 
  with open(filename, 'r', encoding='utf-8') as file:
    return file.read()


def vectorize_text(job_description_text, resume_text):
        job_description_text = load_text_from_file(job_desc)
    #    resume_text = load_text_from_file(resume_text)
        vectorizer = TfidfVectorizer()
        combined_corpus = [job_description_text, resume_text]
        vectors = vectorizer.fit_transform(combined_corpus)
        job_desc_vec = vectors[0:1].toarray()
        resume_text_vec = vectors[1:2].toarray()
        return job_desc_vec, resume_text_vec


def match_resume(job_description_vec, resume_vec):
    Similarity = cosine_similarity(job_description_vec, resume_vec).flatten()[0]
    return Similarity



job_desc = 'job_desc'
#resume_text = 'resume_text.txt'

job_desc_vec, resume_text_vec = vectorize_text(job_desc, extracted_text)
similarity_score = match_resume(job_desc_vec, resume_text_vec)

print("Cosine Similarity between Job Description and Resume:", similarity_score*100)
