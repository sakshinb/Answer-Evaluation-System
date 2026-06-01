"""
NLP Modules Package
Contains all individual NLP analysis modules for the Advanced Grading System
"""

from .semantic_comparator import SemanticComparator
from .grammar_checker import GrammarChecker
from .rubric_engine import RubricEngine
from .fact_checker import FactChecker
from .qa_evaluator import QAEvaluator
from .scoring_aggregator import ScoringAggregator
from .gemini_grader import GeminiGrader
from .groq_grader import GroqGrader
from .numerical_sympy_scorer import NumericalSympyScorer
from .list_set_coverage import ListSetCoverageScorer
from .diagram_ocr_detector import DiagramOCRDetector
from .per_question_rubric import PerQuestionRubric
from .adaptive_weight_engine import AdaptiveWeightEngine

__all__ = [
    'SemanticComparator',
    'GrammarChecker',
    'RubricEngine',
    'FactChecker',
    'QAEvaluator',
    'ScoringAggregator',
    'GeminiGrader',
    'GroqGrader',
    # v3 additions
    'NumericalSympyScorer',
    'ListSetCoverageScorer',
    'DiagramOCRDetector',
    'PerQuestionRubric',
    'AdaptiveWeightEngine',
]

__version__ = '3.0.0'
__author__ = 'Advanced Grading System'
__description__ = (
    'Modular NLP components for comprehensive academic assignment grading. '
    'v3 adds: SymPy equation matching, set-coverage list scoring, '
    'diagram OCR, per-question rubrics, and adaptive weight engine.'
)