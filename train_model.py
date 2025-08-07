import pandas as pd
import numpy as np
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# 1. Load dataset
df = pd.read_csv('resumes.csv')

# 2. Data cleaning function
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\r', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Clean resumes
df['cleaned_resume'] = df['Resume'].apply(clean_text)

# 3. BERT Embedding
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256, padding='max_length')
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Use the [CLS] token embedding
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Get embeddings for all resumes
embeddings = []
for text in tqdm(df['cleaned_resume'], desc='Embedding resumes'):
    try:
        emb = get_bert_embedding(text)
    except Exception as e:
        emb = np.zeros(768)
    embeddings.append(emb)
embeddings = np.vstack(embeddings)

# 4. Train classifier
X = embeddings
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
clf.fit(X_train, y_train)

# 5. Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# 6. Save model and pipeline
joblib.dump(clf, 'resume_classifier.joblib')
joblib.dump(tokenizer, 'bert_tokenizer.joblib')
joblib.dump(bert_model, 'bert_model.joblib')

print('Model and BERT pipeline saved!')