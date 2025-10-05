from flask import Blueprint, render_template, request, current_app  # Add current_app here
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

candidate_bp = Blueprint('candidate', __name__)

# Load all models (same as recruiter)
try:
    # Load sentence transformer
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Sentence Transformer loaded successfully!")
    
    # Load spaCy
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded successfully!")
    
    # Load your trained classifier and BERT models
    try:
        clf = joblib.load('resume_classifier.joblib')
        print("✅ Resume classifier loaded successfully!")
        
        # Load BERT model and tokenizer
        bert_tokenizer = joblib.load('bert_tokenizer.joblib')
        bert_model = joblib.load('bert_model.joblib')
        print("✅ BERT model and tokenizer loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading trained models: {e}")
        clf = None
        bert_tokenizer = None
        bert_model = None
        
except Exception as e:
    print(f"❌ Error loading models: {e}")
    sentence_model = None
    nlp = None
    clf = None
    bert_tokenizer = None
    bert_model = None

def clean_text(text):
    """Text cleaning for ML model (same as training)"""
    text = str(text)
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\r', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def get_bert_embedding(text):
    """Get BERT embedding for text (same as training)"""
    if bert_model is None or bert_tokenizer is None:
        return None
    
    try:
        inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, max_length=256, padding='max_length')
        with torch.no_grad():
            outputs = bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    except Exception as e:
        print(f"Error in BERT embedding: {e}")
        return None

def classify_resume(resume_text):
    """Classify resume using trained model"""
    if clf is None:
        return None
    
    try:
        clean_resume = clean_text(resume_text)
        embedding = get_bert_embedding(clean_resume)
        
        if embedding is not None:
            embedding = embedding.reshape(1, -1)
            predicted_category = clf.predict(embedding)[0]
            probabilities = clf.predict_proba(embedding)[0]
            confidence = np.max(probabilities) * 100
            categories = clf.classes_
            category_scores = {category: round(prob * 100, 2) for category, prob in zip(categories, probabilities)}
            
            return {
                'predicted_category': predicted_category,
                'confidence': confidence,
                'all_categories': category_scores,
                'embedding_used': True
            }
        else:
            return None
            
    except Exception as e:
        print(f"Error in resume classification: {e}")
        return None

def analyze_job_description(job_description):
    """Analyze job description to predict what category it belongs to"""
    if clf is None:
        return None
    
    try:
        clean_job = clean_text(job_description)
        job_embedding = get_bert_embedding(clean_job)
        
        if job_embedding is not None:
            job_embedding = job_embedding.reshape(1, -1)
            job_category = clf.predict(job_embedding)[0]
            job_probabilities = clf.predict_proba(job_embedding)[0]
            job_confidence = np.max(job_probabilities) * 100
            
            return {
                'predicted_category': job_category,
                'confidence': job_confidence,
                'all_probabilities': {cat: round(prob * 100, 2) for cat, prob in zip(clf.classes_, job_probabilities)}
            }
        else:
            return None
            
    except Exception as e:
        print(f"Error in job description analysis: {e}")
        return None

def calculate_category_match(job_analysis, resume_classification):
    """Calculate how well resume category matches job category"""
    if not job_analysis or not resume_classification:
        return 0
    
    if job_analysis['predicted_category'] == resume_classification['predicted_category']:
        match_score = (job_analysis['confidence'] + resume_classification['confidence']) / 2
        return min(100, match_score)
    else:
        job_probs = job_analysis['all_probabilities']
        resume_probs = resume_classification['all_categories']
        common_categories = set(job_probs.keys()) & set(resume_probs.keys())
        if common_categories:
            max_match = max(job_probs[cat] * resume_probs[cat] / 100 for cat in common_categories)
            return min(100, max_match)
        else:
            return 0

def extract_keywords(text):
    """Extract important keywords from text"""
    stop_words = set(stopwords.words('english'))
    words = re.findall(r'\b[a-z][a-z]+\b', text.lower())
    words = [word for word in words if word not in stop_words and len(word) > 2]
    word_freq = Counter(words)
    return [word for word, freq in word_freq.most_common(20)]

def calculate_semantic_similarity(text1, text2):
    """Calculate semantic similarity using sentence transformers"""
    if sentence_model is None:
        return 0.0
    try:
        embeddings = sentence_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return max(0, min(100, similarity * 100))
    except Exception as e:
        print(f"Error in semantic similarity: {e}")
        return 0.0

def calculate_keyword_similarity(text1, text2):
    """Calculate keyword overlap similarity"""
    keywords1 = extract_keywords(text1)
    keywords2 = extract_keywords(text2)
    
    if not keywords1 or not keywords2:
        return 0.0
    
    set1 = set(keywords1)
    set2 = set(keywords2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = intersection / union if union > 0 else 0.0
    return similarity * 100

def advanced_resume_analysis(job_description, resume_text, filename):
    """Main analysis using trained classification model"""
    job_analysis = analyze_job_description(job_description)
    resume_classification = classify_resume(resume_text)
    
    clean_job = clean_text(job_description)
    clean_resume = clean_text(resume_text)
    
    semantic_score = calculate_semantic_similarity(clean_job, clean_resume)
    keyword_score = calculate_keyword_similarity(clean_job, clean_resume)
    
    if job_analysis and resume_classification:
        category_match_score = calculate_category_match(job_analysis, resume_classification)
        final_score = (
            category_match_score * 0.7 +
            semantic_score * 0.2 +
            keyword_score * 0.1
        )
        model_used = True
    else:
        final_score = (
            semantic_score * 0.6 +
            keyword_score * 0.4
        )
        category_match_score = 0
        model_used = False
    
    final_score = max(0, min(100, final_score))
    
    return {
        'filename': filename,
        'final_score': final_score,
        'model_used': model_used,
        'job_analysis': job_analysis,
        'resume_classification': resume_classification,
        'category_match_score': category_match_score,
        'semantic_score': semantic_score,
        'keyword_score': keyword_score,
        'text_preview': resume_text[:200] + "..." if len(resume_text) > 200 else resume_text
    }

def extract_skills(text):
    """Extract skills from text using basic pattern matching"""
    common_skills = {
        'python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'angular', 
        'node', 'docker', 'aws', 'azure', 'machine learning', 'data analysis',
        'project management', 'agile', 'scrum', 'leadership', 'communication'
    }
    
    found_skills = set()
    for skill in common_skills:
        if skill in text.lower():
            found_skills.add(skill)
    
    return found_skills

def generate_resume_suggestions(job_description, resume_text):
    """Generate AI-powered suggestions for resume improvement"""
    suggestions = []
    
    clean_job = clean_text(job_description)
    clean_resume = clean_text(resume_text)
    
    # Extract keywords from both
    job_keywords = set(extract_keywords(clean_job))
    resume_keywords = set(extract_keywords(clean_resume))
    
    # Find missing keywords
    missing_keywords = job_keywords - resume_keywords
    if missing_keywords:
        suggestions.append({
            'type': 'missing_keywords',
            'title': 'Add Important Keywords',
            'description': f'Consider adding these keywords from the job description: {", ".join(list(missing_keywords)[:10])}',
            'priority': 'high'
        })
    
    # Category matching analysis
    job_analysis = analyze_job_description(job_description)
    resume_classification = classify_resume(resume_text)
    
    if job_analysis and resume_classification:
        if job_analysis['predicted_category'] != resume_classification['predicted_category']:
            suggestions.append({
                'type': 'category_mismatch',
                'title': 'Align with Job Category',
                'description': f'Your resume is classified as {resume_classification["predicted_category"]} but the job requires {job_analysis["predicted_category"]}. Consider emphasizing relevant skills for the target role.',
                'priority': 'high'
            })
    
    # Skills gap analysis
    job_skills = extract_skills(clean_job)
    resume_skills = extract_skills(clean_resume)
    missing_skills = job_skills - resume_skills
    if missing_skills:
        suggestions.append({
            'type': 'missing_skills',
            'title': 'Develop Missing Skills',
            'description': f'The job requires these skills that are not prominent in your resume: {", ".join(list(missing_skills)[:8])}',
            'priority': 'medium'
        })
    
    # Content length analysis
    word_count = len(clean_resume.split())
    if word_count < 200:
        suggestions.append({
            'type': 'content_length',
            'title': 'Expand Resume Content',
            'description': 'Your resume seems brief. Consider adding more detailed descriptions of your projects, achievements, and responsibilities.',
            'priority': 'medium'
        })
    elif word_count > 800:
        suggestions.append({
            'type': 'content_length',
            'title': 'Condense Resume',
            'description': 'Your resume might be too long. Consider making it more concise and focused on key achievements.',
            'priority': 'low'
        })
    
    # Experience level analysis
    experience_keywords = ['senior', 'lead', 'manager', 'director', 'head', 'experienced', 'expert']
    job_level = any(keyword in clean_job.lower() for keyword in experience_keywords)
    resume_experience = any(keyword in clean_resume.lower() for keyword in experience_keywords)
    
    if job_level and not resume_experience:
        suggestions.append({
            'type': 'experience_level',
            'title': 'Highlight Leadership Experience',
            'description': 'The job seems to require senior/leadership experience. Consider emphasizing any leadership roles or major projects you have led.',
            'priority': 'medium'
        })
    
    return suggestions

@candidate_bp.route('/candidate', methods=['GET', 'POST'])
def candidate_dashboard():
    """Candidate functionality - single resume suggestions"""
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_file = request.files['resume']
        
        if resume_file.filename:
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], resume_file.filename)  # Use current_app
            resume_file.save(filepath)
            resume_text = extract_resume_text(filepath)
            
            analysis = advanced_resume_analysis(job_description, resume_text, resume_file.filename)
            suggestions = generate_resume_suggestions(job_description, resume_text)
            
            return render_template('candidate_results.html', 
                                 analysis=analysis, 
                                 suggestions=suggestions,
                                 job_description=job_description)
    
    return render_template('candidate_dashboard.html')