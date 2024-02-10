from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

# Function to tokenize the dataset
class MovieReviewDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

# Load the preprocessed reviews and labels
df = pd.read_csv('../../data/processed/preprocessed_reviews_transformers.csv')
y = pd.read_pickle('../../data/processed/labels_transformers.pkl')

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['review'], y, test_size=0.2, random_state=42)

# Initialize the tokenizer and model from the transformers library
model_name = "bert-base-uncased"  # You can replace "bert-base-uncased" with any other model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Adjust num_labels as per your task

# Tokenize the text
train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=512)


# Convert to torch Dataset
train_dataset = MovieReviewDataset(train_encodings, y_train)
test_dataset = MovieReviewDataset(test_encodings, y_test)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory for model predictions and checkpoints
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,   # batch size per device during training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",     # evaluate each `logging_steps`
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=lambda p: {"accuracy": (np.argmax(p.predictions, axis=1) == p.label_ids).mean()},
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()

print(f"Evaluation results: {eval_results}")

# Optionally, save the trained model for later use or deployment
model.save_pretrained('./movie_review_model')
tokenizer.save_pretrained('./movie_review_model')

print("Model training and evaluation complete. Model saved.")
