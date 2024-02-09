from flask import Flask, request, jsonify
import pickle
import re
import spacy
from utils.data_processing import clean_and_lemmatize
app = Flask(__name__)

# Initialize spaCy language model
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Load the trained model
model_path = '../models/logistic_regression_model.pkl'  # Update with your model path
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load the TF-IDF vectorizer
vectorizer_path = '../data/processed/tfidf_vectorizer.pkl'  # Update with your vectorizer path
with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# Function to clean and lemmatize the text
def clean_and_lemmatize(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and numbers (keep only letters)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Lowercase all texts
    text = text.lower()
    # Lemmatize with spaCy
    doc = nlp(text)
    lemmatized = [token.lemma_ for token in doc if not token.is_stop and len(token.text) > 2]
    return ' '.join(lemmatized)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    review = data['review']
    # Preprocess the review
    processed_review = clean_and_lemmatize(review)
    # Vectorize the preprocessed review
    vectorized_review = vectorizer.transform([processed_review])
    # Predict sentiment
    prediction = model.predict(vectorized_review)[0]
    sentiment = 'positive' if prediction == 1 else 'negative'
    return jsonify({'review': review, 'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
