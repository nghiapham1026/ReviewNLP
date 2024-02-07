import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from tqdm.auto import tqdm  # Import tqdm for progress bars

# Configure tqdm to work with pandas apply()
tqdm.pandas()

# Download NLTK stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('../../data/raw/IMDB_Dataset.csv')

# Function to clean the text
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Lowercase all texts
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and stem
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens if not word in set(stopwords.words('english'))]
    return ' '.join(tokens)

# Apply the cleaning function to the review column with progress bar
df['review'] = df['review'].progress_apply(clean_text)

# Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['review']).toarray()

# Convert sentiments to binary labels
y = df['sentiment'].map({'positive': 1, 'negative': 0}).values