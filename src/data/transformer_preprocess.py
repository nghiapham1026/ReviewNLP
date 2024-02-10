import pandas as pd
import re
import spacy
from tqdm.auto import tqdm
import pickle

import utils.data_processing as utils

# Load the dataset
df = pd.read_csv('../../data/raw/IMDB_Dataset.csv')

# Initialize spaCy language model, if you still want to clean or lemmatize
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Configure tqdm to work with pandas apply()
tqdm.pandas()

# Since transformers can handle a wide vocabulary, you might decide not to lemmatize
# Apply the cleaning function to the review column with progress bar
# If lemmatization is needed, ensure your cleaning function includes it
df['review'] = df['review'].progress_apply(utils.clean_text)  # Assume clean_text is your updated function

# Convert sentiments to binary labels
y = df['sentiment'].map({'positive': 1, 'negative': 0}).values

# Save the processed dataframe and labels only
df.to_csv('../../data/processed/preprocessed_reviews_transformers.csv', index=False)

with open('../../data/processed/labels_transformers.pkl', 'wb') as f:
    pickle.dump(y, f)

print("Preprocessing complete and files saved.")
