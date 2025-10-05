# from flask import Flask, render_template, request, redirect, url_for
# import os
# import joblib
# import numpy as np
# import re
# import string
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# import torch
# from nltk.corpus import stopwords
# import nltk
# from utils import extract_resume_text
# import spacy
# from collections import Counter
# import json
# import pandas as pd

# # Download required NLTK data
# try:
#     stopwords.words('english')
# except LookupError:
#     nltk.download('stopwords')

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads/'

# # Ensure upload folder exists
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # Load all models
# try:
#     # Load sentence transformer
#     sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
#     print("✅ Sentence Transformer loaded successfully!")
    
#     # Load spaCy
#     nlp = spacy.load("en_core_web_sm")
#     print("✅ spaCy model loaded successfully!")
    
#     # Load your trained classifier and BERT models
#     try:
#         clf = joblib.load('resume_classifier.joblib')
#         print("✅ Resume classifier loaded successfully!")
        
#         # Load BERT model and tokenizer
#         bert_tokenizer = joblib.load('bert_tokenizer.joblib')
#         bert_model = joblib.load('bert_model.joblib')
#         print("✅ BERT model and tokenizer loaded successfully!")
        
#     except Exception as e:
#         print(f"❌ Error loading trained models: {e}")
#         clf = None
#         bert_tokenizer = None
#         bert_model = None
        
# except Exception as e:
#     print(f"❌ Error loading models: {e}")
#     sentence_model = None
#     nlp = None
#     clf = None
#     bert_tokenizer = None
#     bert_model = None

# def clean_text(text):
#     """Text cleaning for ML model (same as training)"""
#     text = str(text)
#     text = text.lower()
#     text = re.sub(r'\n', ' ', text)
#     text = re.sub(r'\r', ' ', text)
#     text = re.sub(r'\s+', ' ', text)
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     stop_words = set(stopwords.words('english'))
#     text = ' '.join([word for word in text.split() if word not in stop_words])
#     return text

# def get_bert_embedding(text):
#     """Get BERT embedding for text (same as training)"""
#     if bert_model is None or bert_tokenizer is None:
#         return None
    
#     try:
#         inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, max_length=256, padding='max_length')
#         with torch.no_grad():
#             outputs = bert_model(**inputs)
#         # Use the [CLS] token embedding
#         return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
#     except Exception as e:
#         print(f"Error in BERT embedding: {e}")
#         return None

# def classify_resume(resume_text):
#     """Classify resume using trained model"""
#     if clf is None:
#         return None
    
#     try:
#         # Clean the resume text
#         clean_resume = clean_text(resume_text)
        
#         # Get BERT embedding
#         embedding = get_bert_embedding(clean_resume)
        
#         if embedding is not None:
#             # Reshape for single sample
#             embedding = embedding.reshape(1, -1)
            
#             # Get prediction and probabilities
#             predicted_category = clf.predict(embedding)[0]
#             probabilities = clf.predict_proba(embedding)[0]
            
#             # Get confidence score
#             confidence = np.max(probabilities) * 100
            
#             # Get all categories and their probabilities
#             categories = clf.classes_
#             category_scores = {category: round(prob * 100, 2) for category, prob in zip(categories, probabilities)}
            
#             return {
#                 'predicted_category': predicted_category,
#                 'confidence': confidence,
#                 'all_categories': category_scores,
#                 'embedding_used': True
#             }
#         else:
#             return None
            
#     except Exception as e:
#         print(f"Error in resume classification: {e}")
#         return None

# def analyze_job_description(job_description):
#     """Analyze job description to predict what category it belongs to"""
#     if clf is None:
#         return None
    
#     try:
#         # Clean job description
#         clean_job = clean_text(job_description)
        
#         # Get BERT embedding for job description
#         job_embedding = get_bert_embedding(clean_job)
        
#         if job_embedding is not None:
#             # Reshape for single sample
#             job_embedding = job_embedding.reshape(1, -1)
            
#             # Predict category for job description
#             job_category = clf.predict(job_embedding)[0]
#             job_probabilities = clf.predict_proba(job_embedding)[0]
#             job_confidence = np.max(job_probabilities) * 100
            
#             return {
#                 'predicted_category': job_category,
#                 'confidence': job_confidence,
#                 'all_probabilities': {cat: round(prob * 100, 2) for cat, prob in zip(clf.classes_, job_probabilities)}
#             }
#         else:
#             return None
            
#     except Exception as e:
#         print(f"Error in job description analysis: {e}")
#         return None

# def calculate_category_match(job_analysis, resume_classification):
#     """Calculate how well resume category matches job category"""
#     if not job_analysis or not resume_classification:
#         return 0
    
#     # If same category, high score
#     if job_analysis['predicted_category'] == resume_classification['predicted_category']:
#         # Weighted by confidence of both predictions
#         match_score = (job_analysis['confidence'] + resume_classification['confidence']) / 2
#         return min(100, match_score)
#     else:
#         # Different categories - check if there's some overlap in probabilities
#         job_probs = job_analysis['all_probabilities']
#         resume_probs = resume_classification['all_categories']
        
#         # Calculate similarity between probability distributions
#         common_categories = set(job_probs.keys()) & set(resume_probs.keys())
#         if common_categories:
#             # Use the maximum probability product for common categories
#             max_match = max(job_probs[cat] * resume_probs[cat] / 100 for cat in common_categories)
#             return min(100, max_match)
#         else:
#             return 0

# def advanced_resume_analysis(job_description, resume_text, filename):
#     """Main analysis using trained classification model"""
    
#     # Step 1: Analyze job description to understand what category it needs
#     job_analysis = analyze_job_description(job_description)
    
#     # Step 2: Classify the resume
#     resume_classification = classify_resume(resume_text)
    
#     # Step 3: Calculate traditional scores as backup
#     clean_job = clean_text(job_description)
#     clean_resume = clean_text(resume_text)
    
#     semantic_score = calculate_semantic_similarity(clean_job, clean_resume)
#     keyword_score = calculate_keyword_similarity(clean_job, clean_resume)
    
#     # Step 4: Calculate category matching score
#     if job_analysis and resume_classification:
#         category_match_score = calculate_category_match(job_analysis, resume_classification)
        
#         # Primary scoring: Use category matching (70%) + traditional (30%)
#         final_score = (
#             category_match_score * 0.7 +
#             semantic_score * 0.2 +
#             keyword_score * 0.1
#         )
#         model_used = True
#     else:
#         # Fallback: Use only traditional scoring
#         final_score = (
#             semantic_score * 0.6 +
#             keyword_score * 0.4
#         )
#         category_match_score = 0
#         model_used = False
    
#     # Ensure score is between 0-100
#     final_score = max(0, min(100, final_score))
    
#     return {
#         'filename': filename,
#         'final_score': final_score,
#         'model_used': model_used,
#         'job_analysis': job_analysis,
#         'resume_classification': resume_classification,
#         'category_match_score': category_match_score,
#         'semantic_score': semantic_score,
#         'keyword_score': keyword_score,
#         'text_preview': resume_text[:200] + "..." if len(resume_text) > 200 else resume_text
#     }

# # Traditional scoring functions
# def extract_keywords(text):
#     """Extract important keywords from text"""
#     stop_words = set(stopwords.words('english'))
#     words = re.findall(r'\b[a-z][a-z]+\b', text.lower())
#     words = [word for word in words if word not in stop_words and len(word) > 2]
#     word_freq = Counter(words)
#     return [word for word, freq in word_freq.most_common(20)]

# def calculate_semantic_similarity(text1, text2):
#     """Calculate semantic similarity using sentence transformers"""
#     if sentence_model is None:
#         return 0.0
#     try:
#         embeddings = sentence_model.encode([text1, text2])
#         similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
#         return max(0, min(100, similarity * 100))
#     except Exception as e:
#         print(f"Error in semantic similarity: {e}")
#         return 0.0

# def calculate_keyword_similarity(text1, text2):
#     """Calculate keyword overlap similarity"""
#     keywords1 = extract_keywords(text1)
#     keywords2 = extract_keywords(text2)
    
#     if not keywords1 or not keywords2:
#         return 0.0
    
#     set1 = set(keywords1)
#     set2 = set(keywords2)
    
#     intersection = len(set1.intersection(set2))
#     union = len(set1.union(set2))
    
#     similarity = intersection / union if union > 0 else 0.0
#     return similarity * 100

# def process_resumes(job_description, resume_texts):
#     """Process all resumes using the classification model"""
#     results = []
    
#     for resume in resume_texts:
#         analysis = advanced_resume_analysis(
#             job_description, 
#             resume['text'], 
#             resume['filename']
#         )
        
#         # Format for display
#         result_data = {
#             "filename": analysis['filename'],
#             "similarity": round(analysis['final_score'], 1),
#             "text": analysis['text_preview'],
#             "semantic_score": round(analysis['semantic_score'], 1),
#             "keyword_score": round(analysis['keyword_score'], 1),
#             "model_used": analysis['model_used'],
#             "category_match_score": round(analysis['category_match_score'], 1)
#         }
        
#         # Add classification results if available
#         if analysis['model_used']:
#             if analysis['job_analysis']:
#                 result_data["job_category"] = analysis['job_analysis']['predicted_category']
#                 result_data["job_confidence"] = round(analysis['job_analysis']['confidence'], 1)
            
#             if analysis['resume_classification']:
#                 result_data["resume_category"] = analysis['resume_classification']['predicted_category']
#                 result_data["resume_confidence"] = round(analysis['resume_classification']['confidence'], 1)
#                 result_data["all_categories"] = analysis['resume_classification']['all_categories']
        
#         results.append(result_data)
    
#     # Sort by final score (highest first) - PROPER RANKING!
#     results.sort(key=lambda x: x['similarity'], reverse=True)
#     return results

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         job_description = request.form['job_description']
#         resume_files = request.files.getlist('resumes')
#         saved_files = []
#         resume_texts = []
        
#         for file in resume_files:
#             if file.filename:
#                 filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#                 file.save(filepath)
#                 saved_files.append(filepath)
#                 text = extract_resume_text(filepath)
#                 resume_texts.append({'filename': file.filename, 'text': text})
        
#         # Process resumes with classification model
#         results = process_resumes(job_description, resume_texts)
#         return render_template('results.html', results=results, job_description=job_description)
    
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, render_template
import os
from recruiter import recruiter_bp
from candidate import candidate_bp

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Register blueprints
app.register_blueprint(recruiter_bp)
app.register_blueprint(candidate_bp)

@app.route('/')
def main_dashboard():
    """Main dashboard - choose between recruiter and candidate"""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)