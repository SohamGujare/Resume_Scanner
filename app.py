from flask import Flask, render_template, request, redirect, url_for
import os
import joblib
import numpy as np
import re
import string
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
from nltk.corpus import stopwords
import nltk
from utils import extract_resume_text
import spacy
from collections import Counter
import json

# Download required NLTK data
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load advanced models
try:
    # Load sentence transformer for better semantic similarity
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Sentence Transformer loaded successfully!")
    
    # Load spaCy for NER and advanced NLP
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded successfully!")
    
    # Load the trained classifier (if available)
    try:
        clf = joblib.load('resume_classifier.joblib')
        print("Classifier loaded successfully!")
    except:
        clf = None
        print("No classifier found, using advanced scoring only")
        
except Exception as e:
    print(f"Error loading models: {e}")
    sentence_model = None
    nlp = None
    clf = None

def clean_text(text):
    """Advanced text cleaning"""
    text = str(text)
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\r', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # Keep some punctuation for NER
    text = re.sub(r'[^\w\s\.\,\-\&]', ' ', text)
    return text.strip()

def extract_entities(text):
    """Extract named entities (organizations, institutions, skills)"""
    if nlp is None:
        return {'orgs': [], 'skills': []}
    
    doc = nlp(text)
    orgs = [ent.text.lower() for ent in doc.ents if ent.label_ in ['ORG', 'GPE']]
    # Extract potential skills (words in caps, technical terms)
    skills = re.findall(r'\b[A-Z][A-Z\s]+\b', text)
    skills = [skill.strip().lower() for skill in skills if len(skill.strip()) > 2]
    
    return {'orgs': orgs, 'skills': skills}

def extract_keywords(text):
    """Extract important keywords from text"""
    # Remove stopwords and get word frequency
    stop_words = set(stopwords.words('english'))
    words = re.findall(r'\b\w+\b', text.lower())
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Get most common words
    word_freq = Counter(words)
    return [word for word, freq in word_freq.most_common(20)]

def calculate_semantic_similarity(text1, text2):
    """Calculate semantic similarity using sentence transformers"""
    if sentence_model is None:
        return 0.0
    
    try:
        embeddings = sentence_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    except:
        return 0.0

def calculate_keyword_similarity(keywords1, keywords2):
    """Calculate keyword overlap similarity"""
    if not keywords1 or not keywords2:
        return 0.0
    
    set1 = set(keywords1)
    set2 = set(keywords2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

def calculate_entity_similarity(entities1, entities2):
    """Calculate similarity based on named entities"""
    orgs1 = set(entities1.get('orgs', []))
    orgs2 = set(entities2.get('orgs', []))
    
    org_intersection = len(orgs1.intersection(orgs2))
    org_union = len(orgs1.union(orgs2))
    
    org_similarity = org_intersection / org_union if org_union > 0 else 0.0
    
    # Bonus for exact organization matches
    exact_matches = len(orgs1.intersection(orgs2))
    
    return org_similarity + (exact_matches * 0.1)

def advanced_resume_scoring(job_description, resume_text):
    """Advanced multi-layered scoring system"""
    
    # Clean texts
    clean_job = clean_text(job_description)
    clean_resume = clean_text(resume_text)
    
    # 1. Semantic similarity (40% weight)
    semantic_score = calculate_semantic_similarity(clean_job, clean_resume)
    
    # 2. Keyword similarity (30% weight)
    job_keywords = extract_keywords(clean_job)
    resume_keywords = extract_keywords(clean_resume)
    keyword_score = calculate_keyword_similarity(job_keywords, resume_keywords)
    
    # 3. Entity similarity (20% weight)
    job_entities = extract_entities(job_description)
    resume_entities = extract_entities(resume_text)
    entity_score = calculate_entity_similarity(job_entities, resume_entities)
    
    # 4. Exact phrase matching (10% weight)
    exact_phrases = 0
    job_words = set(clean_job.split())
    resume_words = set(clean_resume.split())
    
    # Check for exact word matches (case-insensitive)
    for job_word in job_words:
        if job_word in resume_words:
            exact_phrases += 1
    
    exact_score = exact_phrases / len(job_words) if job_words else 0.0
    
    # Weighted final score
    final_score = (
        semantic_score * 0.4 +
        keyword_score * 0.3 +
        entity_score * 0.2 +
        exact_score * 0.1
    )
    
    return {
        'final_score': final_score,
        'semantic_score': semantic_score,
        'keyword_score': keyword_score,
        'entity_score': entity_score,
        'exact_score': exact_score,
        'job_keywords': job_keywords[:10],
        'resume_keywords': resume_keywords[:10],
        'job_entities': job_entities,
        'resume_entities': resume_entities
    }

def process_resumes(job_description, resume_texts):
    """Process resumes with advanced scoring"""
    results = []
    
    for resume in resume_texts:
        # Get advanced scoring
        scoring = advanced_resume_scoring(job_description, resume['text'])
        
        results.append({
            "filename": resume['filename'],
            "similarity": round(scoring['final_score'], 4),
            "text": resume['text'][:200] + "..." if len(resume['text']) > 200 else resume['text'],
            "semantic_score": round(scoring['semantic_score'], 4),
            "keyword_score": round(scoring['keyword_score'], 4),
            "entity_score": round(scoring['entity_score'], 4),
            "exact_score": round(scoring['exact_score'], 4),
            "job_keywords": scoring['job_keywords'],
            "resume_keywords": scoring['resume_keywords'],
            "job_entities": scoring['job_entities'],
            "resume_entities": scoring['resume_entities']
        })
    
    # Sort by final similarity score (highest first)
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_files = request.files.getlist('resumes')
        saved_files = []
        resume_texts = []
        for file in resume_files:
            if file.filename:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                saved_files.append(filepath)
                # Extract text from the uploaded resume
                text = extract_resume_text(filepath)
                resume_texts.append({'filename': file.filename, 'text': text})
        
        # Process resumes with advanced scoring
        results = process_resumes(job_description, resume_texts)
        return render_template('results.html', results=results, job_description=job_description)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)