import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
import pickle

# Initialize spaCy language model
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Configure tqdm to work with pandas apply()
tqdm.pandas()

# Load the dataset
df = pd.read_csv('../../data/raw/IMDB_Dataset.csv')

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

# Apply the cleaning and lemmatization function to the review column with progress bar
df['review'] = df['review'].progress_apply(clean_and_lemmatize)

# Vectorization (TF-IDF) with updated parameters
vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.7, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['review']).toarray()

# Convert sentiments to binary labels
y = df['sentiment'].map({'positive': 1, 'negative': 0}).values

# Save the processed dataframe, vectorized features, and labels
df.to_csv('../../data/processed/preprocessed_reviews.csv', index=False)

with open('../../data/processed/tfidf_features.pkl', 'wb') as f:
    pickle.dump(X, f)

with open('../../data/processed/labels.pkl', 'wb') as f:
    pickle.dump(y, f)

with open('../../data/processed/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Preprocessing complete and files saved.")
