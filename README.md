# ReviewNLP

ReviewNLP is a project focused on processing and analyzing movie reviews using various NLP (Natural Language Processing) techniques. It aims to classify reviews into positive or negative sentiments and provides insights into the underlying emotions and opinions expressed in the reviews.

## Project Structure

The project is organized into several key directories and files:

- `data/`: Contains raw and processed datasets.
  - `raw/`: Houses the original dataset, `IMDB_Dataset.csv`, used for analysis.
  - `processed/`: Includes preprocessed data and model-related files like `labels.pkl`, `preprocessed_reviews.csv`, and `tfidf_vectorizer.pkl`.
- `models/`: Contains the machine learning models used for sentiment analysis, including Logistic Regression, Random Forest, SVM, and Transformer models.
- `src/`: The source code directory.
  - `data/`: Scripts for data preprocessing (`preprocess_data.py`, `transformer_preprocess.py`).
  - `models/`: Implementation of different machine learning models (`lr_model.py`, `rfc_model.py`, `svm_model.py`, `transformers_model.py`).
  - `utils/`: Utility scripts for data processing and hyperparameter tuning.
  - `app.py`: The Flask application for deploying the model as a web service.
  - `templates/`: Contains HTML templates for the web interface.
- `requirements.txt`: Lists all the dependencies required to run the project.
- `runtime.txt`: Specifies the Python version.
- `Procfile` and `start.sh`: Configuration files for deploying the application.

## Setup
***Note: the root directory is at `/src`***

To set up the project locally, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/nghiapham1026/ReviewNLP.git
   cd ReviewNLP
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Flask application at the root (/src) directory. By default, the script uses Logistic Regression model to train task:
   ```
   python app.py
   ```

## Usage

### Preprocessing the Dataset

To prepare the dataset for training, first navigate to the `src/data` directory. Then, execute the following commands based on the model you plan to use:

- For traditional machine learning models:
  ```
  python preprocess_data.py
  ```
- For deep learning models:
  ```
  python transformer_preprocess.py
  ```

### Training Models

The repository includes several machine learning models for sentiment analysis on the IMDB dataset. Navigate to the `src/models` directory to run the training scripts. Ensure that the appropriate preprocessing script has been executed beforehand.

- **Logistic Regression**:
  ```
  python lr_model.py
  ```
- **Random Forest Classifier**:
  ```
  python rfc_model.py
  ```
- **Support Vector Machine (SVM)**:
  ```
  python svm_model.py
  ```
  **Note:** The SVM and transformer models are computationally intensive. It is recommended to run these models on machines with adequate resources.

- **Transformer Models**:
  ```
  python transformers_model.py
  ```
  Ensure that you have preprocessed the dataset using `transformers_preprocess.py` before training transformer models.