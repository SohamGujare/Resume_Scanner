🎯 ResumePulse - AI Resume Analysis Platform
An advanced AI-powered resume analysis platform that serves both recruiters and job seekers with cutting-edge Natural Language Processing (NLP) and machine learning technologies.

✨ Features
👔 Recruiter Mode
Multi-Resume Analysis: Process and rank multiple resumes simultaneously

Smart Candidate Ranking: AI-powered scoring and prioritization

Category Classification: Automatic resume categorization using BERT embeddings

Detailed Analytics: Comprehensive scoring breakdown with semantic and keyword matching

Batch Processing: Handle multiple PDF/DOCX files efficiently

🎯 Candidate Mode
Personalized Suggestions: AI-powered resume improvement recommendations

Job Matching Score: Real-time compatibility analysis with job descriptions

Skill Gap Analysis: Identify missing skills and keywords

ATS Optimization: Improve resume parsing success rates

Career Alignment: Category matching and experience level analysis

🚀 Technology Stack
Backend: Python, Flask, Flask Blueprints

Machine Learning: BERT Transformers, Scikit-learn, Sentence Transformers

NLP: spaCy, NLTK, Cosine Similarity

File Processing: pdfminer.six, python-docx

Frontend: Modern HTML5, CSS3 with Glassmorphism effects

Styling: Advanced animations, gradient backgrounds, responsive design

🏗️ Project Architecture
text
ResumePulse/
├── app.py                 # Main Flask application with dashboard
├── recruiter.py           # Recruiter functionality blueprint
├── candidate.py           # Candidate functionality blueprint  
├── utils.py              # File processing utilities
├── train_model.py        # BERT model training script
├── requirements.txt      # Python dependencies
├── templates/           # HTML templates
│   ├── index.html      # Main dashboard (role selection)
│   ├── recruiter_dashboard.html    # Recruiter interface
│   ├── candidate_dashboard.html    # Candidate interface
│   ├── results.html    # Recruiter results page
│   └── candidate_results.html    # Candidate analysis results
├── uploads/             # File upload directory
└── models/             # Trained ML models
    ├── resume_classifier.joblib
    ├── bert_model.joblib
    └── bert_tokenizer.joblib
📋 Prerequisites
Python 3.7 or higher

pip (Python package installer)

🛠️ Installation & Setup
Clone the repository

bash
git clone <your-repository-url>
cd ResumePulse
Create virtual environment (Recommended)

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

bash
pip install -r requirements.txt
Download spaCy model

bash
python -m spacy download en_core_web_sm
Train the AI model (First time setup)

bash
python train_model.py
Run the application

bash
python app.py
Access the platform
Navigate to http://localhost:5000 in your browser

🎯 How to Use
For Recruiters 👔
Select "Recruiter Mode" from the main dashboard

Enter the detailed job description

Upload multiple resumes (PDF/DOCX format)

Get AI-powered rankings with detailed scoring breakdown

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
Text Extraction: PDF/DOCX to clean text conversion

BERT Embeddings: 768-dimensional vector representations

Classification: Logistic regression on BERT features

Similarity Scoring: Cosine similarity and keyword overlap

Suggestion Generation: Pattern-based improvement recommendations

🎨 UI/UX Features
Dual-Mode Dashboard: Clean role selection interface

Modern Dark Theme: Professional gradient backgrounds with animations

Responsive Design: Mobile-first approach with seamless cross-device experience

Interactive Elements: Hover effects, loading animations, smooth transitions

Glassmorphism Effects: Advanced CSS backdrop filters and transparency

Particle Animations: Dynamic background elements for enhanced engagement

⚙️ Configuration
Model Customization
Retrain the classification model with your data:

bash
python train_model.py
Scoring Weights
Adjust algorithm weights in recruiter.py and candidate.py:

Category matching importance

Semantic similarity factors

Keyword scoring ratios

📊 Performance Metrics
Processing Speed: 2-4 seconds per resume analysis

Accuracy: High-precision BERT-based categorization

Scalability: Modular architecture supporting multiple concurrent users

File Support: Robust PDF and DOCX parsing

🔒 Data Privacy
Local processing - no data sent to external servers

Secure file handling with automatic cleanup

Encrypted storage for sensitive information

Privacy-first design for both recruiters and candidates

ResumePulse - Where AI meets career success for both recruiters and job seekers 🚀

Made with ❤️ using Advanced AI & Machine Learning
