from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Inisialisasi pipeline untuk question-answering
qa_pipeline = pipeline(
    "question-answering",
    model="Rifky/Indobert-QA",
    tokenizer="Rifky/Indobert-QA"
)

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    context = data.get('context')
    question = data.get('question')
    
    # Validasi input
    if not context or not question:
        return jsonify({'error': 'Both context and question are required'}), 400

    # Menggunakan model untuk menjawab pertanyaan
    result = qa_pipeline({
        'context': context,
        'question': question
    })
    
    # Mengembalikan hasil dalam format JSON
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
