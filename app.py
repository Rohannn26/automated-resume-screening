from flask import Flask, request, jsonify, render_template
from text_extraction import extract_text_from_pdf, vectorize_text, match_resume
#from vectorization import vectorize_text, match_resume
#from matching import match_resume

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_resume():
    job_description_file = request.files['job-description']
    job_description_text = job_description_file.read().decode('utf-8')
    job_role = request.form['job-role']
    
    resume_files = request.files.getlist('resume')
    result = []

    for resume_file in resume_files:
        resume_text = ""
        if resume_file.filename.endswith('.pdf'):
            resume_text = extract_text_from_pdf(resume_file)
        elif resume_file.filename.endswith('.docx'):
            pass
    
        job_description_vec, resume_vec = vectorize_text(job_description_text, resume_text)
        similarity_score = match_resume(job_description_vec, resume_vec)
    
        result.append({
            "name": resume_file.filename,
            "score": similarity_score*100   
        })

    result.sort(key=lambda x: x["score"], reverse=True)
    
    return jsonify({"ranked_candidates": result, "job_role": job_role})

if __name__ == '__main__':
    app.run(debug=True)
