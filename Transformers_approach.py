import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from transformers import (BertTokenizer, BertForSequenceClassification, 
                          DistilBertTokenizer, DistilBertForSequenceClassification,
                          RobertaTokenizer, RobertaForSequenceClassification,
                          XLMRobertaTokenizer, XLMRobertaForSequenceClassification,
                          Trainer, TrainingArguments)
from torch.utils.data import Dataset
import torch
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

# Download NLTK data
nltk.download('stopwords')

# Function to preprocess text
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.strip()  # Remove leading and trailing spaces
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove words with length 1 or 2
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    words = [stemmer.stem(word) for word in words]  # Apply stemming
    return ' '.join(words)

# Load training and validation data
train_df = pd.read_csv("./train.csv", header=None, names=["abs_text", "label_text"])
val_df = pd.read_csv("./validation.csv", header=None, names=["abs_text", "label_text"])

# Apply preprocessing to training and validation data
train_df['cleaned_abstract'] = train_df['abs_text'].apply(preprocess_text)
val_df['cleaned_abstract'] = val_df['abs_text'].apply(preprocess_text)

# Split validation data into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(val_df['cleaned_abstract'], val_df['label_text'], test_size=0.2, random_state=42)

# Concatenate training and validation sets
train_texts = train_df['cleaned_abstract'].tolist()
train_labels = train_df['label_text'].tolist()
val_texts = X_val.tolist()
val_labels = y_val.tolist()
test_texts = X_test.tolist()
test_labels = y_test.tolist()

# Convert string labels to numerical labels
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
val_labels = label_encoder.transform(val_labels)
test_labels = label_encoder.transform(test_labels)

# Define tokenizers for different models
tokenizer_dict = {
    'bert': BertTokenizer.from_pretrained('bert-base-uncased'),
    'distilbert': DistilBertTokenizer.from_pretrained('distilbert-base-uncased'),
    'roberta': RobertaTokenizer.from_pretrained('roberta-base'),
    'xlm-roberta': XLMRobertaTokenizer.from_pretrained('xlm-roberta-base'),
}

# Define models and their configurations
models = {
    'bert': (BertForSequenceClassification, 'bert-base-uncased'),
    'distilbert': (DistilBertForSequenceClassification, 'distilbert-base-uncased'),
    'roberta': (RobertaForSequenceClassification, 'roberta-base'),
    'xlm-roberta': (XLMRobertaForSequenceClassification, 'xlm-roberta-base'),
}

# Function to tokenize texts
def tokenize_function(texts, tokenizer):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=128)

# Dataset class for handling tokenized data
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Function to evaluate a model checkpoint
def evaluate_checkpoint(checkpoint_dir, model_class, tokenizer, test_texts, test_labels):
    # Load model from checkpoint
    model = model_class.from_pretrained(checkpoint_dir)
    
    # Tokenize test texts
    test_encodings = tokenize_function(test_texts, tokenizer)
    
    # Prepare test dataset
    test_dataset = TextDataset(test_encodings, test_labels)
    
    # Training arguments for evaluation
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        per_device_eval_batch_size=32,
        logging_dir='./logs',
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
    )
    
    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args
    )
    
    # Predict on test set using the trained model
    predictions = trainer.predict(test_dataset)
    y_test_pred = np.argmax(predictions.predictions, axis=1)
    
    # Calculate test F1 score and return
    test_f1 = f1_score(test_labels, y_test_pred, average='weighted')
    return test_f1, y_test_pred

# Store results for each model
results = []

for model_name, (model_class, pretrained_model) in models.items():
    print(f"Training and evaluating {model_name}...")
    
    # Tokenize data for training and validation sets
    train_encodings = tokenize_function(train_texts, tokenizer_dict[model_name])
    val_encodings = tokenize_function(val_texts, tokenizer_dict[model_name])
    
    # Create datasets for training and validation
    train_dataset = TextDataset(train_encodings, train_labels)
    val_dataset = TextDataset(val_encodings, val_labels)
    
    # Initialize the model with pretrained weights
    model = model_class.from_pretrained(pretrained_model, num_labels=7)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f'./results_{model_name}',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'./logs_{model_name}',
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        fp16=True,
        learning_rate=2e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
    )
    
    # Create Trainer instance for training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Train the model
    trainer.train()

    # Evaluate the best checkpoints found during training on the test set
    best_f1 = 0
    best_report = ""
    best_checkpoint = ""

    for checkpoint in ['checkpoint-525', 'checkpoint-1050', 'checkpoint-1575']:
        checkpoint_dir = f'./results_{model_name}/{checkpoint}'
        
        if not os.path.exists(checkpoint_dir):
            print(f"Checkpoint directory {checkpoint_dir} does not exist.")
            continue
        
        # Evaluate the model checkpoint
        test_f1, y_test_pred = evaluate_checkpoint(checkpoint_dir, model_class, tokenizer_dict[model_name], test_texts, test_labels)
        
        # Track the best performing checkpoint
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_checkpoint = checkpoint_dir
            best_report = classification_report(test_labels, y_test_pred)
    
    # Print and store results
    print(f"Best checkpoint for {model_name}: {best_checkpoint} with F1 score: {best_f1}")
    
    results.append({
        'Model': model_name,
        'Best Checkpoint': best_checkpoint,
        'Test F1': best_f1,
        'Classification Report': best_report
    })

# Display results as a DataFrame
results_df = pd.DataFrame(results)
print("\nComparison of Models:")
print(results_df)
results_df.to_excel('LLM_comparision_results.csv')