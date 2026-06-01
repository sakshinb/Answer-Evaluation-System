"""
Flask Web Application for Advanced Grading System
Provides a user-friendly interface for PDF upload, grading, and feedback
"""

import os
import json
import uuid
import tempfile
from datetime import datetime
from typing import Dict, List, Optional
import logging

from dotenv import load_dotenv
load_dotenv()  # loads .env file automatically

from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pandas as pd

# Import grading system components
from grading_system import AdvancedGradingSystem
from minimal_preprocessor import MinimalPreprocessor
from modules.groq_grader import GroqGrader
from modules.trocr_ocr import extract_text as trocr_extract_text
from database import db, GradingSession as DBSession, UploadedFile as DBFile, GradingResult as DBResult, Question as DBQuestion

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = 'advanced_grading_system_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# SQLite database config
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///gradesavvy.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff', 'tif'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

grading_sessions = {}

_GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
_groq_grader_instance: Optional['GroqGrader'] = None


# Create DB tables on first run
def init_db():
    with app.app_context():
        db.create_all()
        logger.info("✅ Database tables created / verified")


def get_groq_grader(api_key: str = None) -> 'GroqGrader':
    """Return the cached GroqGrader instance, reinitialising only if the API key changes."""
    global _groq_grader_instance, _GROQ_API_KEY
    key = api_key or _GROQ_API_KEY
    if _groq_grader_instance is None or key != _GROQ_API_KEY:
        logger.info("Initialising GroqGrader singleton...")
        _GROQ_API_KEY = key
        _groq_grader_instance = GroqGrader(key)
    return _groq_grader_instance


class GradingSession:
    """Manages a grading session with uploaded files and configuration"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.uploaded_files = []
        self.questions = []
        self.rubric = {}
        self.grading_config = {
            'semantic_weight': 0.25,
            'rubric_weight': 0.20,
            'grammar_weight': 0.25,
            'factual_weight': 0.15,
            'completeness_weight': 0.15
        }
        self.results = []
        self.status = 'initialized'

    def add_file(self, filename: str, filepath: str, file_type: str):
        self.uploaded_files.append({
            'filename': filename,
            'filepath': filepath,
            'file_type': file_type,
            'uploaded_at': datetime.now().isoformat()
        })

    def set_questions(self, questions: List[str]):
        self.questions = questions

    def set_rubric(self, rubric: Dict):
        self.rubric = rubric

    def update_config(self, config: Dict):
        self.grading_config.update(config)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _pdf_text_layer_fallback(pdf_path: str) -> str:
    """Extract embedded text layer from a PDF (for digitally-typed PDFs)."""
    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception:
        pass
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception:
        pass
    return ""


def extract_text_from_file(file_path: str, file_type: str) -> str:
    """
    Extract text from any supported file.

    For plain .txt files: read directly.
    For .pdf: try embedded text layer first, then fall back to OCR.
    For all image types: use PaddleOCR with TrOCR fallback.
    """
    # BUG FIX 1: was indented incorrectly, causing IndentationError at startup.
    file_type = file_type.lower().lstrip(".")

    if file_type == "txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # BUG FIX 2: _pdf_text_layer_fallback was never called.
    # For PDFs, try the embedded text layer first (fast, lossless for typed PDFs).
    if file_type == "pdf":
        text = _pdf_text_layer_fallback(file_path)
        if len(text.strip()) >= 50:
            logger.info("PDF text layer extracted (%d chars) from %s", len(text), os.path.basename(file_path))
            return text
        logger.info("PDF text layer too short (%d chars); falling back to OCR.", len(text.strip()))

    # PDF (no text layer) and all image formats → OCR
    logger.info("OCR extraction: %s (%s)", os.path.basename(file_path), file_type)
    text = ""
    try:
        text = paddle_extract_text(file_path)
        if len(text.strip()) < 50:
            logger.warning("PaddleOCR weak extraction (%d chars). Falling back to TrOCR.", len(text.strip()))
            text = trocr_extract_text(file_path)
    except Exception as e:
        logger.error("PaddleOCR failed: %s. Falling back to TrOCR.", e)
        try:
            text = trocr_extract_text(file_path)
        except Exception as e2:
            logger.error("TrOCR also failed: %s", e2)

    if text.strip():
        logger.info("OCR extracted %d chars from %s", len(text), os.path.basename(file_path))
        return text

    # BUG FIX 3: original function had no return statement on the success path,
    # so it always returned None, causing downstream AttributeError on .strip() etc.
    logger.warning("All extraction methods returned empty text for %s", os.path.basename(file_path))
    return ""


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/new_session', methods=['POST'])
def new_session():
    session_id = str(uuid.uuid4())
    grading_sessions[session_id] = GradingSession(session_id)
    # Persist to DB
    db_session = DBSession(id=session_id, status='initialized')
    db.session.add(db_session)
    db.session.commit()
    return jsonify({'success': True, 'session_id': session_id, 'message': 'New grading session created'})


@app.route('/upload_file/<session_id>', methods=['POST'])
def upload_file(session_id):
    if session_id not in grading_sessions:
        return jsonify({'success': False, 'message': 'Invalid session ID'})

    session = grading_sessions[session_id]

    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file provided'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'File type not allowed. Supported: PDF, TXT, JPG, PNG, WEBP, BMP, TIFF.'})

    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(filepath)

    file_type = filename.rsplit('.', 1)[1].lower()
    session.add_file(unique_filename, filepath, file_type)

    # Persist file record to DB
    db_file = DBFile(session_id=session_id, filename=unique_filename, filepath=filepath, file_type=file_type)
    db.session.add(db_file)
    db.session.commit()

    return jsonify({
        'success': True,
        'message': f'File {filename} uploaded successfully',
        'filename': unique_filename,
        'files_count': len(session.uploaded_files)
    })


@app.route('/set_questions/<session_id>', methods=['POST'])
def set_questions(session_id):
    if session_id not in grading_sessions:
        return jsonify({'success': False, 'message': 'Invalid session ID'})

    session = grading_sessions[session_id]
    data = request.get_json() or {}
    questions = data.get('questions', [])

    if not questions:
        return jsonify({'success': False, 'message': 'No questions provided'})

    session.set_questions(questions)
    # Persist questions to DB
    DBQuestion.query.filter_by(session_id=session_id).delete()
    for i, q in enumerate(questions):
        db.session.add(DBQuestion(session_id=session_id, text=q, order=i))
    db.session.commit()
    return jsonify({'success': True, 'message': f'{len(questions)} question(s) set', 'questions_count': len(questions)})


@app.route('/set_model_answer/<session_id>', methods=['POST'])
def set_model_answer(session_id):
    if session_id not in grading_sessions:
        return jsonify({'success': False, 'message': 'Invalid session ID'})
    session = grading_sessions[session_id]
    data = request.get_json() or {}
    model_answer = data.get('model_answer', '').strip()
    if not model_answer:
        return jsonify({'success': False, 'message': 'No model answer provided'})
    session.model_answer = model_answer
    return jsonify({'success': True, 'message': 'Model answer saved'})


@app.route('/set_rubric_manual/<session_id>', methods=['POST'])
def set_rubric_manual(session_id):
    if session_id not in grading_sessions:
        return jsonify({'success': False, 'message': 'Invalid session ID'})
    session = grading_sessions[session_id]
    data = request.get_json() or {}
    rubric = data.get('rubric', {})
    session.set_rubric(rubric)
    return jsonify({'success': True, 'message': 'Rubric saved'})


@app.route('/start_grading/<session_id>', methods=['POST'])
def start_grading(session_id):
    logger.info(f"Starting HYBRID NLP + AI grading for session: {session_id}")

    if session_id not in grading_sessions:
        return jsonify({'success': False, 'message': 'Invalid session ID'})

    session = grading_sessions[session_id]
    logger.info(f"Session found. Files: {len(session.uploaded_files)}, Questions: {len(session.questions)}")

    if not session.uploaded_files:
        return jsonify({'success': False, 'message': 'No files uploaded'})
    if not session.questions:
        return jsonify({'success': False, 'message': 'No questions provided'})

    try:
        session.status = 'processing'

        groq_grader = get_groq_grader()

        if hasattr(session, 'model_answer') and session.model_answer:
            model_answer = session.model_answer
            logger.info("Using manually provided model answer.")
        else:
            logger.info("Generating AI Model Answer via Groq...")
            model_answer = groq_grader.generate_model_answer(
                session.questions,
                session.rubric if session.rubric else {}
            )

        if session.rubric and session.rubric.get('criteria'):
            ai_rubric = session.rubric
            logger.info("Using manually provided rubric.")
        else:
            logger.info("Generating AI Rubric via Groq...")
            ai_rubric = groq_grader.generate_rubric_from_gemini(
                session.questions,
                max_marks=10
            )

        grader_weights = {
            'semantic':     session.grading_config.get('semantic_weight', 0.25),
            'rubric':       session.grading_config.get('rubric_weight', 0.20),
            'grammar':      session.grading_config.get('grammar_weight', 0.25),
            'factual':      session.grading_config.get('factual_weight', 0.15),
            'completeness': session.grading_config.get('completeness_weight', 0.15)
        }

        grader = AdvancedGradingSystem(grader_weights)
        preprocessor = MinimalPreprocessor()

        results = []

        for file_info in session.uploaded_files:
            try:
                logger.info(f"Processing file: {file_info['filename']}")

                student_text = extract_text_from_file(
                    file_info['filepath'], file_info['file_type']
                )

                preprocessed = preprocessor.preprocess_text(student_text)
                cleaned_text = preprocessed['cleaned_text']

                result = grader.grade_assignment(
                    student_text=cleaned_text,
                    reference_text=model_answer,
                    rubric=ai_rubric,
                    student_id=file_info['filename'],
                    assignment_id=f"assignment_{len(results)+1}",
                    question=" ".join(session.questions) if session.questions else ""
                )

                # Blend Groq LLM comparison into the final score.
                # NLP metrics (semantic/rubric/factual) are weak against LLM model answers
                # because of length mismatch and paraphrasing. Groq's direct comparison
                # is far more accurate, so we give it 50% weight when available.
                groq_score_blended = result.final_score
                groq_similarity = None
                groq_used = False
                factual_score = result.factual_score
                if not groq_grader.use_mock:
                    try:
                        rubric_for_cmp = {
                            c['name']: c['description']
                            for c in ai_rubric.get('criteria', [])
                        } if ai_rubric.get('criteria') else {}
                        groq_result = groq_grader.compare_with_model(
                            cleaned_text, model_answer, rubric_for_cmp
                        )
                        raw_groq = groq_result.similarity_score * 100
                        # Direct linear scale — no artificial floor.
                        # Groq returns 0-100; use it as-is capped at 100.
                        calibrated_groq = min(raw_groq, 100.0)
                        groq_score_blended = result.final_score * 0.50 + calibrated_groq * 0.50
                        groq_similarity = round(groq_result.similarity_score, 3)
                        groq_used = groq_result.groq_used

                        # Use Groq's content_accuracy score to boost factual score.
                        # NLP fact checker is unreliable against LLM model answers;
                        # Groq understands factual correctness semantically.
                        groq_content_accuracy = groq_result.detailed_scores.get('content_accuracy')
                        if groq_content_accuracy is not None:
                            groq_factual = min(float(groq_content_accuracy) * 100, 100.0)
                            # Blend: 40% NLP fact checker + 60% Groq content accuracy
                            factual_score = result.factual_score * 0.40 + groq_factual * 0.60

                        # Merge Groq feedback
                        result.strengths   = list(dict.fromkeys(result.strengths   + groq_result.feedback['strengths']))
                        result.weaknesses  = list(dict.fromkeys(result.weaknesses  + groq_result.feedback['weaknesses']))
                        result.suggestions = list(dict.fromkeys(result.suggestions + groq_result.feedback['suggestions']))
                        logger.info(f"Groq blend: nlp={result.final_score:.1f}, groq={calibrated_groq:.1f}, blended={groq_score_blended:.1f}, factual={factual_score:.1f}")
                    except Exception as ge:
                        logger.warning(f"Groq blend failed for {file_info['filename']}: {ge}")

                results.append({
                    'filename': file_info['filename'],
                    'student_id': result.student_id,
                    'final_score': round(groq_score_blended, 2),
                    'semantic_score': round(result.semantic_score, 2),
                    'grammar_score': round(result.grammar_score, 2),
                    'factual_score': round(factual_score, 2),
                    'completeness_score': round(result.completeness_score, 2),
                    'rubric_score': round(result.rubric_score, 2),
                    'confidence_score': round(result.confidence_score, 3),
                    'word_count': getattr(result, 'word_count', 0),
                    'sentence_count': getattr(result, 'sentence_count', 0),
                    'technical_terms_found': getattr(result, 'technical_terms_found', 0),
                    'grammar_issues': getattr(result, 'grammar_issues', 0),
                    'strengths': result.strengths,
                    'weaknesses': result.weaknesses,
                    'suggestions': result.suggestions,
                    'model_answer_used': model_answer,
                    'rubric_used': ai_rubric,
                    'student_info': preprocessed['student_info'],
                    'groq_similarity': groq_similarity,
                    'groq_used': groq_used,
                })

            except Exception as e:
                logger.error(f"Error processing {file_info['filename']}: {e}")
                results.append({
                    'filename': file_info['filename'],
                    'processing_failed': True,
                    'error': str(e)
                })

        session.results = results
        session.status = 'completed'

        # Persist results to DB
        db_sess = DBSession.query.get(session_id)
        if db_sess:
            db_sess.status = 'completed'
        else:
            db.session.add(DBSession(id=session_id, status='completed'))

        for r in results:
            if not r.get('processing_failed'):
                db.session.add(DBResult(
                    session_id         = session_id,
                    filename           = r['filename'],
                    student_name       = r.get('student_info', {}).get('name', 'Unknown'),
                    final_score        = r['final_score'],
                    semantic_score     = r['semantic_score'],
                    rubric_score       = r['rubric_score'],
                    grammar_score      = r['grammar_score'],
                    factual_score      = r['factual_score'],
                    completeness_score = r['completeness_score'],
                    confidence_score   = r['confidence_score'],
                    word_count         = r.get('word_count', 0),
                    sentence_count     = r.get('sentence_count', 0),
                    grammar_issues     = r.get('grammar_issues', 0),
                    strengths          = json.dumps(r.get('strengths', [])),
                    weaknesses         = json.dumps(r.get('weaknesses', [])),
                    suggestions        = json.dumps(r.get('suggestions', [])),
                    model_answer_used  = r.get('model_answer_used', ''),
                    rubric_used        = json.dumps(r.get('rubric_used', {})),
                    groq_similarity    = r.get('groq_similarity'),
                ))
        db.session.commit()
        logger.info("✅ Results saved to database")

        logger.info("Hybrid grading completed successfully.")

        return jsonify({
            'success': True,
            'message': f'Hybrid grading completed for {len(results)} files',
            'ai_model_answer_generated': True,
            'ai_rubric_generated': True,
            'results_count': len(results)
        })

    except Exception as e:
        session.status = 'error'
        logger.error(f"Hybrid grading error: {e}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/get_results/<session_id>')
def get_results(session_id):
    logger.info(f"Getting results for session: {session_id}")
    logger.info(f"Available sessions: {list(grading_sessions.keys())}")

    if session_id not in grading_sessions:
        return jsonify({'success': False, 'message': f'Invalid session ID. Available sessions: {len(grading_sessions)}'})

    session = grading_sessions[session_id]
    logger.info(f"Session status: {session.status}")

    if session.status != 'completed':
        return jsonify({'success': False, 'message': f'Grading not completed. Status: {session.status}'})

    if session.results:
        scores = [r.get('final_score', 0) for r in session.results if not r.get('processing_failed')]
        summary = {
            'total_files': len(session.results),
            'successful_gradings': len(scores),
            'average_score': round(sum(scores) / len(scores), 2) if scores else 0,
            'highest_score': max(scores) if scores else 0,
            'lowest_score': min(scores) if scores else 0
        }
    else:
        summary = {
            'total_files': 0, 'successful_gradings': 0,
            'average_score': 0, 'highest_score': 0, 'lowest_score': 0
        }

    return jsonify({
        'success': True,
        'results': session.results,
        'summary': summary,
        'session_info': {
            'session_id': session_id,
            'created_at': session.created_at.isoformat(),
            'questions_count': len(session.questions),
            'files_count': len(session.uploaded_files)
        }
    })


@app.route('/export_results/<session_id>')
def export_results(session_id):
    if session_id not in grading_sessions:
        return jsonify({'success': False, 'message': 'Invalid session ID'})

    session = grading_sessions[session_id]
    if not session.results:
        return jsonify({'success': False, 'message': 'No results to export'})

    df_data = []
    for result in session.results:
        if not result.get('processing_failed'):
            df_data.append({
                'filename': result['filename'],
                'student_name': result.get('student_info', {}).get('name', 'Unknown'),
                'final_score': result['final_score'],
                'semantic_score': result['semantic_score'],
                'rubric_score': result['rubric_score'],
                'grammar_score': result['grammar_score'],
                'factual_score': result['factual_score'],
                'completeness_score': result['completeness_score'],
                'confidence_score': result['confidence_score'],
                # BUG FIX 4 (continued): use .get() with defaults so export never crashes
                # even if a result was produced by an older code path.
                'word_count': result.get('word_count', 0),
                'sentence_count': result.get('sentence_count', 0),
                'technical_terms': result.get('technical_terms_found', 0),
                'grammar_issues': result.get('grammar_issues', 0),
                'strengths': '; '.join(result.get('strengths', [])),
                'weaknesses': '; '.join(result.get('weaknesses', [])),
                'suggestions': '; '.join(result.get('suggestions', []))
            })

    if df_data:
        df = pd.DataFrame(df_data)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"grading_results_{session_id[:8]}_{timestamp}.csv"
        filepath = os.path.join(RESULTS_FOLDER, filename)
        df.to_csv(filepath, index=False)
        return send_file(filepath, as_attachment=True, download_name=filename)

    return jsonify({'success': False, 'message': 'No valid results to export'})


@app.route('/start_gemini_grading/<session_id>', methods=['POST'])
def start_gemini_grading(session_id):
    logger.info(f"Starting Gemini-enhanced grading for session: {session_id}")

    if session_id not in grading_sessions:
        return jsonify({'success': False, 'message': 'Invalid session ID'})

    session = grading_sessions[session_id]

    if not session.uploaded_files:
        return jsonify({'success': False, 'message': 'No files uploaded'})
    if not session.questions:
        return jsonify({'success': False, 'message': 'No questions provided'})

    try:
        session.status = 'processing'
        logger.info("Starting Gemini-enhanced grading process...")

        data = request.get_json() or {}
        groq_api_key = data.get('groq_api_key') or os.environ.get('GROQ_API_KEY', '')

        groq_grader = get_groq_grader(groq_api_key)

        grader_weights = {
            'semantic':     session.grading_config.get('semantic_weight', 0.25),
            'rubric':       session.grading_config.get('rubric_weight', 0.20),
            'grammar':      session.grading_config.get('grammar_weight', 0.25),
            'factual':      session.grading_config.get('factual_weight', 0.15),
            'completeness': session.grading_config.get('completeness_weight', 0.15)
        }

        grader = AdvancedGradingSystem(grader_weights, gemini_grader_instance=groq_grader)
        preprocessor = MinimalPreprocessor()

        rubric_criteria = {
            'accuracy': 'Correct and factual information',
            'completeness': 'Comprehensive coverage of topics',
            'clarity': 'Clear and well-organized explanation',
            'understanding': 'Demonstrates deep understanding'
        }

        results = []

        for file_info in session.uploaded_files:
            try:
                logger.info(f"Processing file: {file_info['filename']}")
                student_text = extract_text_from_file(
                    file_info['filepath'], file_info['file_type']
                )

                logger.info(f"Extracted text length: {len(student_text)}")
                preprocessed = preprocessor.preprocess_text(student_text)
                cleaned_text = preprocessed['cleaned_text']
                logger.info(f"Cleaned text length: {len(cleaned_text)}")

                result = grader.grade_assignment_with_gemini(
                    student_text=cleaned_text,
                    questions=session.questions,
                    rubric_criteria=rubric_criteria,
                    student_id=file_info['filename'],
                    assignment_id=f"gemini_assignment_{len(results)+1}"
                )

                logger.info(f"Gemini grading completed. Final score: {result.final_score}")

                results.append({
                    'filename': file_info['filename'],
                    'student_id': result.student_id,
                    'final_score': round(result.final_score, 2),
                    'semantic_score': round(result.semantic_score, 2),
                    'rubric_score': round(result.rubric_score, 2),
                    'grammar_score': round(result.grammar_score, 2),
                    'factual_score': round(result.factual_score, 2),
                    'completeness_score': round(result.completeness_score, 2),
                    'confidence_score': round(result.confidence_score, 3),
                    'word_count': result.word_count,
                    'sentence_count': result.sentence_count,
                    'technical_terms_found': result.technical_terms_found,
                    'grammar_issues': result.grammar_issues,
                    'strengths': result.strengths,
                    'weaknesses': result.weaknesses,
                    'suggestions': result.suggestions,
                    'processing_time': round(result.processing_time, 2),
                    'student_info': preprocessed['student_info'],
                    'gemini_model_answer': getattr(result, 'gemini_model_answer', ''),
                    'gemini_similarity': round(getattr(result, 'gemini_similarity', 0), 3),
                    'gemini_used': getattr(result, 'gemini_used', False),
                    'gemini_detailed_scores': getattr(result, 'gemini_detailed_scores', {}),
                    'gemini_rubric': getattr(result, 'gemini_rubric', {})
                })

            except Exception as e:
                logger.error(f"Error processing file {file_info['filename']}: {e}")
                results.append({
                    'filename': file_info['filename'],
                    'error': str(e),
                    'final_score': 0,
                    'processing_failed': True
                })

        session.results = results
        session.status = 'completed'
        logger.info(f"Gemini grading completed. Results: {len(results)}")

        return jsonify({
            'success': True,
            'message': f'Groq-enhanced grading completed for {len(results)} files',
            'results_count': len(results),
            'groq_used': grader.use_groq
        })

    except Exception as e:
        session.status = 'error'
        logger.error(f"Gemini grading error: {e}")
        return jsonify({'success': False, 'message': f'Gemini grading failed: {str(e)}'})


@app.route('/generate_model_answer', methods=['POST'])
def generate_model_answer():
    try:
        data = request.get_json() or {}
        api_key = data.get("api_key") or data.get("groq_api_key") or os.environ.get('GROQ_API_KEY', '')
        questions = data.get('questions', [])
        rubric_criteria = data.get('rubric_criteria', {})

        if not questions:
            return jsonify({'success': False, 'message': 'No questions provided'})

        grader = get_groq_grader(api_key)
        model_answer = grader.generate_model_answer(questions, rubric_criteria)

        return jsonify({
            'success': True,
            'model_answer': model_answer,
            'groq_used': not grader.use_mock,
            'questions_count': len(questions)
        })

    except Exception as e:
        logger.error(f"Model answer generation error: {e}")
        return jsonify({'success': False, 'message': f'Model answer generation failed: {str(e)}'})


@app.route('/generate_rubric', methods=['POST'])
def generate_rubric():
    try:
        data = request.get_json() or {}
        api_key = data.get("api_key") or data.get("groq_api_key") or os.environ.get('GROQ_API_KEY', '')
        questions = data.get('questions', [])
        max_marks = int(data.get('max_marks', 10))

        if not questions:
            return jsonify({'success': False, 'message': 'No questions provided'})

        grader = get_groq_grader(api_key)
        rubric = grader.generate_rubric(questions, max_marks)

        return jsonify({
            'success': True,
            'rubric': rubric,
            'groq_used': not grader.use_mock,
            'questions_count': len(questions)
        })

    except Exception as e:
        logger.error(f"Rubric generation error: {e}")
        return jsonify({'success': False, 'message': f'Rubric generation failed: {str(e)}'})


@app.route('/test_groq_connection', methods=['POST'])
def test_groq_connection():
    try:
        data = request.get_json() or {}
        api_key = data.get("api_key") or os.environ.get('GROQ_API_KEY', '')
        groq_grader = get_groq_grader(api_key)
        test_result = groq_grader.test_connection()
        return jsonify(test_result)
    except Exception as e:
        return jsonify({'success': False, 'message': f'Connection test failed: {str(e)}', 'groq_available': False})


@app.route('/grading_modes')
def grading_modes():
    return jsonify({
        'success': True,
        'modes': [
            {
                'name': 'Standard Grading',
                'description': 'Multi-dimensional analysis with modular NLP components',
                'endpoint': '/start_grading',
                'features': ['Semantic analysis', 'Grammar checking', 'Rubric scoring', 'Factual verification']
            },
            {
                'name': 'Groq-Enhanced Grading',
                'description': 'AI-powered grading with Groq-generated model answers (llama-3.3-70b)',
                'endpoint': '/start_gemini_grading',
                'features': ['AI model answers', 'Advanced comparison', 'Enhanced feedback', 'Higher accuracy']
            }
        ]
    })


# BUG FIX 5: session_status was missing its @app.route decorator — the endpoint was unreachable.
@app.route('/session_status/<session_id>')
def session_status(session_id):
    if session_id not in grading_sessions:
        return jsonify({'success': False, 'message': 'Invalid session ID'})
    session = grading_sessions[session_id]
    return jsonify({
        'success': True,
        'status': session.status,
        'files_count': len(session.uploaded_files),
        'questions_count': len(session.questions),
        'has_rubric': bool(session.rubric),
        'results_count': len(session.results) if session.results else 0
    })


@app.route('/list_sessions')
def list_sessions():
    sessions_info = []
    for session_id, session in grading_sessions.items():
        sessions_info.append({
            'session_id': session_id,
            'status': session.status,
            'files_count': len(session.uploaded_files),
            'questions_count': len(session.questions),
            'results_count': len(session.results) if session.results else 0,
            'created_at': session.created_at.isoformat()
        })
    return jsonify({'success': True, 'total_sessions': len(grading_sessions), 'sessions': sessions_info})


@app.route('/debug_session/<session_id>')
def debug_session(session_id):
    if session_id not in grading_sessions:
        return jsonify({'success': False, 'message': 'Session not found'})
    session = grading_sessions[session_id]
    return jsonify({
        'success': True,
        'session_id': session_id,
        'status': session.status,
        'uploaded_files_count': len(session.uploaded_files),
        'questions_count': len(session.questions),
        'results_count': len(session.results) if session.results else 0,
        'has_results': bool(session.results),
        'results_sample': session.results[:2] if session.results else None,
        'created_at': session.created_at.isoformat()
    })


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json() or {}
        message = data.get('message', '').strip()
        history = data.get('history', [])

        if not message:
            return jsonify({'success': False, 'message': 'No message provided'})

        grader = get_groq_grader()
        if grader.use_mock:
            return jsonify({'success': True, 'reply': "I'm Savvy! The Groq API key isn't configured yet, but I can tell you: upload PDF/TXT student answer files, enter your questions, set max marks, and click Grade. Need help with anything else?"})

        system_prompt = """You are Savvy, a friendly and helpful AI assistant for GradeSavvy AI — an intelligent answer evaluation platform.

You help users with:
- How to use the grading system (upload files, enter questions, set marks, choose grading mode)
- Understanding grading results (scores, model answers, rubrics, strengths/weaknesses)
- Explaining the two grading modes: Standard NLP and Groq AI Enhanced
- What file formats are supported (PDF and TXT, up to 16MB)
- How scores are calculated (semantic similarity, grammar, rubric, factual accuracy, completeness)
- Exporting results as CSV
- General questions about AI-powered grading

Keep responses concise, friendly, and helpful. Use bullet points when listing steps. If asked something unrelated to the platform, politely redirect to platform-related help."""

        messages = [{"role": "system", "content": system_prompt}]
        for h in history[-10:]:
            if h.get('role') in ('user', 'assistant') and h.get('content'):
                messages.append({"role": h['role'], "content": h['content']})
        messages.append({"role": "user", "content": message})

        resp = grader.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.6,
            max_tokens=512,
        )
        reply = resp.choices[0].message.content.strip()
        return jsonify({'success': True, 'reply': reply})

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/enhanced_grading')
def enhanced_grading():
    return render_template('enhanced_grading_interface.html')


@app.route('/grading_interface')
def grading_interface():
    return render_template('grading_interface.html')


@app.route('/results_viewer')
def results_viewer():
    return render_template('results_viewer.html')


@app.route('/gemini_tools')
def gemini_tools():
    return render_template('gemini_tools.html')


@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'message': 'File too large. Maximum size is 16MB.'}), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'success': False, 'message': 'Internal server error occurred.'}), 500


if __name__ == '__main__':
    init_db()
    print("Starting Advanced Grading System Web Interface...")
    print("Database: gradesavvy.db (SQLite)")
    print("Access the application at: http://localhost:5000")
    print("Upload folder:", UPLOAD_FOLDER)
    print("Results folder:", RESULTS_FOLDER)
    app.run(debug=True, host='0.0.0.0', port=5000)


@app.route('/history')
def history():
    """Return all past grading results from the database."""
    try:
        results = DBResult.query.order_by(DBResult.graded_at.desc()).limit(200).all()
        return jsonify({
            'success': True,
            'count': len(results),
            'results': [r.to_dict() for r in results]
        })
    except Exception as e:
        logger.error(f"History fetch error: {e}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/history/session/<session_id>')
def history_by_session(session_id):
    """Return all results for a specific session from the database."""
    try:
        results = DBResult.query.filter_by(session_id=session_id).order_by(DBResult.graded_at.desc()).all()
        return jsonify({
            'success': True,
            'session_id': session_id,
            'count': len(results),
            'results': [r.to_dict() for r in results]
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
