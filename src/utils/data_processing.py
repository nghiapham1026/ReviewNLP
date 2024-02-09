import re
import spacy

# Initialize spaCy language model
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

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
