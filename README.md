# 🎯 ResumePulse - AI Resume Analysis Platform

An advanced AI-powered resume analysis platform that serves both recruiters and job seekers using cutting-edge **Natural Language Processing (NLP)** and **Machine Learning (ML)** technologies.

---

## ✨ Features

### 👔 Recruiter Mode
- **Multi-Resume Analysis**: Process and rank multiple resumes simultaneously  
- **Smart Candidate Ranking**: AI-powered scoring and prioritization  
- **Category Classification**: Automatic resume categorization using BERT embeddings  
- **Detailed Analytics**: Comprehensive scoring breakdown with semantic and keyword matching  
- **Batch Processing**: Efficient handling of multiple PDF/DOCX files  

### 🎯 Candidate Mode
- **Personalized Suggestions**: AI-powered resume improvement recommendations  
- **Job Matching Score**: Real-time compatibility analysis with job descriptions  
- **Skill Gap Analysis**: Identify missing skills and keywords  
- **ATS Optimization**: Improve resume parsing success rates  
- **Career Alignment**: Category matching and experience level analysis  

---

## 🚀 Technology Stack

- **Backend**: Python, Flask, Flask Blueprints  
- **Machine Learning**: BERT Transformers, Scikit-learn, Sentence Transformers  
- **NLP**: spaCy, NLTK, Cosine Similarity  
- **File Processing**: pdfminer.six, python-docx  
- **Frontend**: HTML5, CSS3 (Glassmorphism effects)  
- **Styling**: Advanced animations, gradient backgrounds, responsive design  

---

## 🏗️ Project Architecture

ResumePulse/
├── app.py # Main Flask application with dashboard
├── recruiter.py # Recruiter functionality blueprint
├── candidate.py # Candidate functionality blueprint
├── utils.py # File processing utilities
├── train_model.py # BERT model training script
├── requirements.txt # Python dependencies
├── templates/ # HTML templates
│ ├── index.html # Main dashboard (role selection)
│ ├── recruiter_dashboard.html
│ ├── candidate_dashboard.html
│ ├── results.html # Recruiter results page
│ └── candidate_results.html # Candidate analysis results
├── uploads/ # File upload directory
└── models/ # Trained ML models
├── resume_classifier.joblib
├── bert_model.joblib
└── bert_tokenizer.joblib


---

## 📋 Prerequisites

- Python 3.7 or higher  
- `pip` (Python package installer)  

---

## 🛠️ Installation & Setup

```bash
# Clone the repository
git clone <your-repository-url>
cd ResumePulse

# Create virtual environment (Recommended)
python -m venv venv
# Activate venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Train the AI model (First-time setup)
python train_model.py

# Run the application
python app.py

Open your browser and navigate to: http://localhost:5000

🎯 How to Use
For Recruiters 👔

Select "Recruiter Mode" from the main dashboard

Enter the detailed job description

Upload multiple resumes (PDF/DOCX)

Receive AI-powered rankings with detailed scoring breakdown

Review categorized candidates with confidence scores

For Candidates 🎯

Select "Candidate Mode" from the main dashboard

Paste your target job description

Upload your resume

Receive personalized improvement suggestions

Get skill gap analysis and keyword optimization tips

🔬 Technical Implementation
Multi-Layered AI Analysis

Recruiter Scoring Algorithm:

Category Matching (70%): BERT-based resume classification

Semantic Similarity (20%): Sentence transformer embeddings

Keyword Matching (10%): TF-IDF and exact term matching

Candidate Suggestion Engine:

Missing keyword identification

Skill gap analysis

Category alignment scoring

Content length optimization

Experience level matching

Machine Learning Pipeline

Text Extraction: PDF/DOCX → Clean text

BERT Embeddings: 768-dimensional vector representations

Classification: Logistic regression on BERT features

Similarity Scoring: Cosine similarity & keyword overlap

Suggestion Generation: Pattern-based improvement recommendations

🎨 UI/UX Features

Dual-Mode Dashboard: Clean role selection interface

Modern Dark Theme: Gradient backgrounds with animations

Responsive Design: Mobile-first, cross-device support

Interactive Elements: Hover effects, loading animations, smooth transitions

Glassmorphism Effects: CSS backdrop filters & transparency

Particle Animations: Dynamic background elements

⚙️ Configuration
Model Customization

Retrain the classification model with your own data:

python train_model.py

Scoring Weights

Adjust algorithm weights in recruiter.py and candidate.py:

Category matching importance

Semantic similarity factors

Keyword scoring ratios

📊 Performance Metrics

Processing Speed: 2–4 seconds per resume

Accuracy: High-precision BERT-based categorization

Scalability: Modular architecture supporting concurrent users

File Support: Robust PDF and DOCX parsing

🔒 Data Privacy

Local processing – no external server required

Secure file handling with automatic cleanup

Encrypted storage for sensitive data

Privacy-first design for recruiters & candidates

💖 Built With

Python, Flask, NLP, Machine Learning, HTML5, CSS3

❤️ Advanced AI & ML techniques
