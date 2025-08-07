# 🤖 AI Resume Screening Tool

An advanced AI-powered resume screening and ranking system that uses Natural Language Processing (NLP) to automatically evaluate and shortlist resumes based on job descriptions.

## ✨ Features

- **Advanced NLP Processing**: Uses Sentence Transformers (SBERT) for semantic similarity
- **Multi-layered Scoring System**: Combines semantic similarity, keyword matching, entity recognition, and exact phrase matching
- **Modern Web Interface**: Beautiful dark-themed UI with animations and responsive design
- **File Support**: Handles PDF and DOCX resume formats
- **Detailed Analysis**: Provides comprehensive scoring breakdown and keyword extraction
- **Real-time Processing**: Fast and efficient resume analysis with progress indicators

## 🚀 Technology Stack

- **Backend**: Python, Flask
- **NLP**: Sentence Transformers, spaCy, NLTK
- **Machine Learning**: Scikit-learn, BERT embeddings
- **File Processing**: pdfminer.six, python-docx
- **Frontend**: HTML5, CSS3, JavaScript
- **Styling**: Modern dark theme with glassmorphism effects

## 📋 Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-resume-scanner.git
   cd ai-resume-scanner
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Train the model** (First time setup)
   ```bash
   python train_model.py
   ```

## 🎯 Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:5000`

3. **Upload and analyze**
   - Enter a detailed job description
   - Upload multiple resumes (PDF/DOCX format)
   - Click "Analyze & Rank" to get results

## 📊 How It Works

### Multi-layered Scoring System

The tool uses a sophisticated scoring algorithm that combines:

1. **Semantic Similarity (40%)**: Uses SBERT embeddings to understand meaning and context
2. **Keyword Matching (30%)**: Identifies and matches important terms and skills
3. **Entity Recognition (20%)**: Extracts and matches organizations, locations, and named entities
4. **Exact Phrase Matching (10%)**: Finds precise matches for specific requirements

### Processing Pipeline

1. **Text Extraction**: Converts PDF/DOCX files to plain text
2. **Text Preprocessing**: Cleans and normalizes text data
3. **Feature Extraction**: Generates embeddings and extracts entities
4. **Similarity Calculation**: Computes multi-dimensional similarity scores
5. **Ranking**: Sorts resumes by final composite score

## 📁 Project Structure

```
ai-resume-scanner/
├── app.py                 # Main Flask application
├── train_model.py         # Model training script
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
├── resumes.csv          # Training dataset
├── templates/           # HTML templates
│   ├── index.html      # Main upload page
│   └── results.html    # Results display page
├── static/             # Static assets (CSS, JS)
└── README.md          # Project documentation
```

## 🎨 UI Features

- **Dark Modern Theme**: Professional dark interface with cyan accents
- **Animated Background**: Subtle particle animations and gradient effects
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Interactive Elements**: Hover effects, progress bars, and smooth transitions
- **Loading Animations**: Visual feedback during processing

## 🔧 Configuration

### Model Training

The system uses a pre-trained model for resume categorization. To retrain:

1. Update `resumes.csv` with your training data
2. Run `python train_model.py`
3. The model will be saved as `resume_classifier.joblib`

### Customization

- Modify scoring weights in `app.py` (lines with `advanced_resume_scoring` function)
- Adjust UI colors in CSS files
- Add new file formats in `utils.py`

## 📈 Performance

- **Processing Speed**: ~2-3 seconds per resume
- **Accuracy**: High precision through multi-layered scoring
- **Scalability**: Handles multiple resumes simultaneously
- **Memory Efficient**: Optimized for large document processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## 🙏 Acknowledgments

- **Sentence Transformers**: For semantic similarity calculations
- **spaCy**: For named entity recognition
- **Flask**: For the web framework
- **Google Fonts**: For the beautiful typography



**Made with ❤️ using Advanced AI & NLP Technologies** 