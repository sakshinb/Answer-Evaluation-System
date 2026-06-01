"""
database.py
-----------
SQLAlchemy models for GradeSavvy AI.
Uses SQLite by default (gradesavvy.db in the project folder).
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()


class GradingSession(db.Model):
    __tablename__ = 'grading_sessions'

    id         = db.Column(db.String(36), primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    status     = db.Column(db.String(20), default='initialized')

    questions = db.relationship('Question',      backref='session', lazy=True, cascade='all, delete-orphan')
    files     = db.relationship('UploadedFile',  backref='session', lazy=True, cascade='all, delete-orphan')
    results   = db.relationship('GradingResult', backref='session', lazy=True, cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'session_id':      self.id,
            'status':          self.status,
            'created_at':      self.created_at.isoformat(),
            'questions_count': len(self.questions),
            'files_count':     len(self.files),
            'results_count':   len(self.results),
        }


class Question(db.Model):
    __tablename__ = 'questions'

    id         = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id = db.Column(db.String(36), db.ForeignKey('grading_sessions.id'), nullable=False)
    text       = db.Column(db.Text, nullable=False)
    order      = db.Column(db.Integer, default=0)


class UploadedFile(db.Model):
    __tablename__ = 'uploaded_files'

    id          = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id  = db.Column(db.String(36), db.ForeignKey('grading_sessions.id'), nullable=False)
    filename    = db.Column(db.String(255), nullable=False)
    filepath    = db.Column(db.String(500), nullable=False)
    file_type   = db.Column(db.String(10))
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)


class GradingResult(db.Model):
    __tablename__ = 'grading_results'

    id                 = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id         = db.Column(db.String(36), db.ForeignKey('grading_sessions.id'), nullable=False)
    filename           = db.Column(db.String(255))
    student_name       = db.Column(db.String(255))
    final_score        = db.Column(db.Float)
    semantic_score     = db.Column(db.Float)
    rubric_score       = db.Column(db.Float)
    grammar_score      = db.Column(db.Float)
    factual_score      = db.Column(db.Float)
    completeness_score = db.Column(db.Float)
    confidence_score   = db.Column(db.Float)
    word_count         = db.Column(db.Integer, default=0)
    sentence_count     = db.Column(db.Integer, default=0)
    grammar_issues     = db.Column(db.Integer, default=0)
    strengths          = db.Column(db.Text)
    weaknesses         = db.Column(db.Text)
    suggestions        = db.Column(db.Text)
    model_answer_used  = db.Column(db.Text)
    rubric_used        = db.Column(db.Text)
    groq_similarity    = db.Column(db.Float)
    graded_at          = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id':                 self.id,
            'session_id':         self.session_id,
            'filename':           self.filename,
            'student_name':       self.student_name,
            'final_score':        self.final_score,
            'semantic_score':     self.semantic_score,
            'rubric_score':       self.rubric_score,
            'grammar_score':      self.grammar_score,
            'factual_score':      self.factual_score,
            'completeness_score': self.completeness_score,
            'confidence_score':   self.confidence_score,
            'word_count':         self.word_count,
            'sentence_count':     self.sentence_count,
            'grammar_issues':     self.grammar_issues,
            'strengths':          json.loads(self.strengths   or '[]'),
            'weaknesses':         json.loads(self.weaknesses  or '[]'),
            'suggestions':        json.loads(self.suggestions or '[]'),
            'model_answer_used':  self.model_answer_used,
            'rubric_used':        json.loads(self.rubric_used or '{}'),
            'groq_similarity':    self.groq_similarity,
            'graded_at':          self.graded_at.isoformat(),
        }
