"""
Advanced Grading System - Main Controller
Orchestrates all NLP modules for comprehensive assignment grading
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

# Import all NLP modules
from modules.semantic_comparator import SemanticComparator
from modules.grammar_checker import GrammarChecker
from modules.rubric_engine import RubricEngine
from modules.fact_checker import FactChecker
from modules.qa_evaluator import QAEvaluator
from modules.scoring_aggregator import ScoringAggregator
from modules.groq_grader import GroqGrader
from modules.question_type_classifier import RoutedScorer, QuestionTypeClassifier
from modules.adaptive_weight_engine import AdaptiveWeightEngine
from modules.diagram_ocr_detector import DiagramOCRDetector
from modules.per_question_rubric import PerQuestionRubric

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GradingResult:
    """Complete grading result structure"""
    student_id: str
    assignment_id: str
    timestamp: str
    
    # Scores (0-100)
    semantic_score: float
    rubric_score: float
    grammar_score: float
    factual_score: float
    completeness_score: float
    final_score: float
    
    # Analysis details
    word_count: int
    sentence_count: int
    technical_terms_found: int
    grammar_issues: int
    
    # Feedback
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    
    # Metadata
    processing_time: float
    confidence_score: float


class AdvancedGradingSystem:
    """Main grading system that orchestrates all NLP modules"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 groq_api_key: Optional[str] = None,
                 gemini_grader_instance=None):
        """Initialize the grading system with all modules"""
        
        self.semantic_comparator = SemanticComparator()
        self.grammar_checker = GrammarChecker()
        self.rubric_engine = RubricEngine()
        self.fact_checker = FactChecker()
        self.qa_evaluator = QAEvaluator()
        self.scoring_aggregator = ScoringAggregator(weights)

        # Accept old gemini_grader_instance param for backwards compat, but wrap it
        if gemini_grader_instance is not None and hasattr(gemini_grader_instance, '_chat'):
            self.groq_grader = gemini_grader_instance
        else:
            self.groq_grader = GroqGrader(groq_api_key)
        self.use_groq = not self.groq_grader.use_mock

        # Question-type classifier and routed scorer
        self.classifier   = QuestionTypeClassifier()
        self.routed_scorer = RoutedScorer(
            groq_client       = self.groq_grader.client if not self.groq_grader.use_mock else None,
            groq_model        = "llama-3.3-70b-versatile",
            use_groq_fallback = self.use_groq,
        )

        # v3 additions
        self.adaptive_weight_engine = AdaptiveWeightEngine()
        self.diagram_ocr_detector   = DiagramOCRDetector()
        
        logger.info(f"✅ Advanced Grading System initialized "
                    f"(Groq: {'enabled' if self.use_groq else 'mock mode'})")

    def grade_assignment(self,
                         student_text: str,
                         reference_text: str,
                         rubric: Optional[Dict] = None,
                         student_id: str = "unknown",
                         assignment_id: str = "unknown",
                         question: str = "") -> GradingResult:
        """Grade a single assignment using all modules (no Gemini)"""
        from modules.question_type_classifier import NUMERICAL, CODE, LIST_TYPE

        start_time = datetime.now()
        logger.info(f"Starting grading for student: {student_id}")

        # Detect question type so we can use the right scorer for semantic
        q_type = self.classifier.classify(question) if question else self.classifier.classify(reference_text[:200])
        logger.info(f"Detected question type: {q_type}")

        # For numerical/code/list answers SBERT cosine similarity is meaningless
        # (matrices, equations, code snippets have near-zero text similarity to prose).
        # Use the routed type-aware scorer instead and wrap it as semantic_analysis.
        if q_type in (NUMERICAL, CODE, LIST_TYPE):
            routed = self.routed_scorer.score(question or reference_text[:200], student_text, reference_text)
            semantic_analysis = {
                "score": routed.score,
                "phrase_overlap": routed.score / 100,
                "detailed_scores": routed.detail,
                "coverage_analysis": {},
                "statistics": {"student_concepts": [], "reference_concepts": []},
            }
            logger.info(f"Using routed scorer ({q_type}): score={routed.score:.1f}")
        else:
            semantic_analysis = self.semantic_comparator.calculate_semantic_similarity(student_text, reference_text)

        # Decide whether to use AI rubric or static rubric
        if rubric is not None:
            logger.info("Using AI-generated rubric for evaluation...")
            rubric_analysis = self.rubric_engine.analyze_rubric_coverage(
                student_text,
                reference_text,
                rubric=rubric
            )
        else:
            logger.info("Using default static rubric...")
            rubric_analysis = self.rubric_engine.analyze_rubric_coverage(
                student_text,
                reference_text
            )

        # For numerical/code, rubric keyword matching is also unreliable.
        # If rubric score is near zero but semantic (routed) is decent, boost rubric
        # using the routed score as a proxy so it doesn't drag the final down.
        if q_type in (NUMERICAL, CODE) and rubric_analysis['score'] < 20 and semantic_analysis['score'] > 30:
            rubric_analysis = dict(rubric_analysis)
            rubric_analysis['score'] = round(semantic_analysis['score'] * 0.80, 2)
            logger.info(f"Rubric score boosted for {q_type} answer: {rubric_analysis['score']:.1f}")
        grammar_analysis      = self.grammar_checker.analyze_grammar_and_style(student_text)
        factual_analysis      = self.fact_checker.verify_factual_accuracy(student_text, reference_text)
        completeness_analysis = self.qa_evaluator.analyze_completeness(student_text, reference_text)

        # Apply adaptive weights based on question type so grammar doesn't dominate
        # numerical/code answers where writing style is irrelevant.
        answer_has_code = "```" in student_text or "def " in student_text or "class " in student_text
        is_native = self.adaptive_weight_engine.detect_native_speaker(student_text)
        word_count = len(student_text.split())
        adaptive_weights = self.adaptive_weight_engine.compute_weights(
            question_type    = q_type,
            is_native        = is_native,
            answer_length    = word_count,
            answer_has_code  = answer_has_code,
            base_weights     = self.scoring_aggregator.weights.copy(),
        )
        original_weights = self.scoring_aggregator.weights.copy()
        self.scoring_aggregator.weights = adaptive_weights

        aggregated_result = self.scoring_aggregator.aggregate_scores(
            semantic_analysis, rubric_analysis, grammar_analysis,
            factual_analysis, completeness_analysis
        )

        self.scoring_aggregator.weights = original_weights
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = GradingResult(
            student_id=student_id,
            assignment_id=assignment_id,
            timestamp=datetime.now().isoformat(),
            semantic_score=semantic_analysis['score'],
            rubric_score=rubric_analysis['score'],
            grammar_score=grammar_analysis['score'],
            factual_score=factual_analysis['score'],
            completeness_score=completeness_analysis['score'],
            final_score=aggregated_result['final_score'],
            word_count=grammar_analysis.get('vocabulary_analysis', {}).get('total_words', 0),
            sentence_count=grammar_analysis.get('sentence_analysis', {}).get('total_sentences', 0),
            technical_terms_found=len(semantic_analysis.get('statistics', {}).get('student_concepts', [])),
            grammar_issues=grammar_analysis.get('total_errors', 0),
            strengths=aggregated_result['feedback']['strengths'],
            weaknesses=aggregated_result['feedback']['weaknesses'],
            suggestions=aggregated_result['feedback']['suggestions'],
            processing_time=processing_time,
            confidence_score=aggregated_result['confidence_score']
        )
        
        logger.info(f"Grading completed for {student_id}: {result.final_score:.1f}/100")
        return result

    def grade_assignment_typed(self,
                               student_text: str,
                               reference_text: str,
                               question: str,
                               rubric: Optional[Dict] = None,
                               student_id: str = "unknown",
                               assignment_id: str = "unknown") -> GradingResult:
        """
        Grade a single question-answer pair using the QuestionTypeClassifier.

        The routed specialised score replaces the semantic score in the pipeline:
          - factual    -> _FactualScorer    (entity + number overlap)
          - conceptual -> _ConceptualScorer (concept coverage + depth)
          - analytical -> _AnalyticalScorer (argument + evidence + counterpoints)
          - numerical  -> _NumericalScorer  (sympy / float comparison)
          - code       -> _CodeScorer       (AST similarity)
          - list       -> _ListScorer       (set coverage)

        All other modules (grammar, rubric, factual, completeness) still run
        so the rest of the score breakdown is unchanged.
        """
        from modules.question_type_classifier import NUMERICAL, CODE

        start_time = datetime.now()
        logger.info(f"[Typed] Grading student={student_id}, q='{question[:60]}...'")

        # ── Routed type-aware score ──────────────────────────────────────────
        routed = self.routed_scorer.score(question, student_text, reference_text)
        logger.info(f"[Typed] question_type={routed.question_type}, routed_score={routed.score:.1f}")

        # ── v3: Adaptive weights via AdaptiveWeightEngine ────────────────────
        answer_has_code = "```" in student_text or "def " in student_text or "class " in student_text
        is_native = self.adaptive_weight_engine.detect_native_speaker(student_text)
        word_count = len(student_text.split())

        adaptive_weights = self.adaptive_weight_engine.compute_weights(
            question_type   = routed.question_type,
            is_native       = is_native,
            answer_length   = word_count,
            answer_has_code = answer_has_code,
            base_weights    = self.scoring_aggregator.weights.copy(),
        )
        logger.info(f"[Typed] adaptive_weights={adaptive_weights}, native={is_native}")

        # Temporarily apply adaptive weights
        original_weights = self.scoring_aggregator.weights.copy()
        self.scoring_aggregator.weights = adaptive_weights

        if rubric is not None:
            rubric_analysis = self.rubric_engine.analyze_rubric_coverage(student_text, reference_text, rubric=rubric)
        else:
            rubric_analysis = self.rubric_engine.analyze_rubric_coverage(student_text, reference_text)

        grammar_analysis      = self.grammar_checker.analyze_grammar_and_style(student_text)
        factual_analysis      = self.fact_checker.verify_factual_accuracy(student_text, reference_text)
        completeness_analysis = self.qa_evaluator.analyze_completeness(student_text, reference_text)

        # Use routed score as the semantic_analysis score
        routed_as_semantic = {
            "score": routed.score,
            "phrase_overlap": routed.score / 100,    # used by confidence calc
            "detailed_scores": routed.detail,
            "coverage_analysis": {},
            "statistics": {"student_concepts": [], "reference_concepts": []},
        }

        aggregated = self.scoring_aggregator.aggregate_scores(
            routed_as_semantic, rubric_analysis, grammar_analysis,
            factual_analysis, completeness_analysis
        )

        # Restore original weights after scoring
        self.scoring_aggregator.weights = original_weights

        processing_time = (datetime.now() - start_time).total_seconds()

        result = GradingResult(
            student_id=student_id,
            assignment_id=assignment_id,
            timestamp=datetime.now().isoformat(),
            semantic_score=routed.score,          # routed type-aware score
            rubric_score=rubric_analysis["score"],
            grammar_score=grammar_analysis["score"],
            factual_score=factual_analysis["score"],
            completeness_score=completeness_analysis["score"],
            final_score=aggregated["final_score"],
            word_count=grammar_analysis.get("vocabulary_analysis", {}).get("total_words", 0),
            sentence_count=grammar_analysis.get("sentence_analysis", {}).get("total_sentences", 0),
            technical_terms_found=0,
            grammar_issues=grammar_analysis.get("total_errors", 0),
            strengths=aggregated["feedback"]["strengths"],
            weaknesses=aggregated["feedback"]["weaknesses"],
            suggestions=aggregated["feedback"]["suggestions"],
            processing_time=processing_time,
            confidence_score=aggregated["confidence_score"],
        )
        # Attach type metadata for the caller
        result.question_type       = routed.question_type
        result.routed_score_detail  = routed.detail
        result.used_groq_fallback   = routed.used_groq_fallback
        result.adaptive_weights     = adaptive_weights
        result.is_native_speaker    = is_native

        logger.info(f"[Typed] Completed {student_id}: final={result.final_score:.1f}, type={routed.question_type}")
        return result

    def grade_assignments_batch(self, assignments: List[Dict]) -> List[GradingResult]:
        """Grade multiple assignments"""
        results = []
        logger.info(f"Starting batch grading for {len(assignments)} assignments")
        for i, assignment in enumerate(assignments):
            logger.info(f"Grading assignment {i+1}/{len(assignments)}")
            result = self.grade_assignment(
                student_text=assignment['student_text'],
                reference_text=assignment['reference_text'],
                student_id=assignment.get('student_id', f'student_{i+1}'),
                assignment_id=assignment.get('assignment_id', f'assignment_{i+1}')
            )
            results.append(result)
        logger.info(f"Batch grading completed: {len(results)} assignments processed")
        return results


    def export_results(self, results: List[GradingResult], output_file: str):
        """Export results to CSV"""
        data = []
        for result in results:
            data.append({
                'student_id': result.student_id,
                'assignment_id': result.assignment_id,
                'final_score': result.final_score,
                'semantic_score': result.semantic_score,
                'rubric_score': result.rubric_score,
                'grammar_score': result.grammar_score,
                'factual_score': result.factual_score,
                'completeness_score': result.completeness_score,
                'word_count': result.word_count,
                'sentence_count': result.sentence_count,
                'technical_terms_found': result.technical_terms_found,
                'grammar_issues': result.grammar_issues,
                'confidence_score': result.confidence_score,
                'processing_time': result.processing_time,
                'strengths': '; '.join(result.strengths),
                'weaknesses': '; '.join(result.weaknesses),
                'suggestions': '; '.join(result.suggestions),
                'timestamp': result.timestamp
            })
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        logger.info(f"✅ Results exported to {output_file}")

    def generate_detailed_report(self, result: GradingResult) -> Dict[str, Any]:
        """Generate a detailed report for a single result"""
        return self.scoring_aggregator.generate_detailed_report(
            {
                'final_score': result.final_score,
                'confidence_score': result.confidence_score,
                'individual_scores': {
                    'semantic_score': result.semantic_score,
                    'rubric_score': result.rubric_score,
                    'grammar_score': result.grammar_score,
                    'factual_score': result.factual_score,
                    'completeness_score': result.completeness_score
                },
                'feedback': {
                    'strengths': result.strengths,
                    'weaknesses': result.weaknesses,
                    'suggestions': result.suggestions
                },
                'timestamp': result.timestamp
            },
            result.student_id,
            result.assignment_id
        )

    def grade_assignment_with_gemini(self,
                                     student_text: str,
                                     questions: List[str],
                                     rubric_criteria: Dict[str, str],
                                     student_id: str = "unknown",
                                     assignment_id: str = "unknown") -> GradingResult:
        """
        Full Gemini-enhanced grading pipeline:
        1. Generate rubric from Gemini
        2. Generate model answer from Gemini
        3. Run standard NLP pipeline
        4. Run Gemini comparison against model answer
        5. Combine scores + attach all Gemini data to result
        """
        start_time = datetime.now()
        logger.info(f"Starting Gemini-enhanced grading for student: {student_id}")

        # ── Step 1: Generate rubric from Gemini ──────────────────────────────
        total_marks = 100
        logger.info("Generating rubric from Groq...")
        gemini_rubric = self.groq_grader.generate_rubric_from_gemini(questions, max_marks=total_marks)
        logger.info(f"Rubric generated: {len(gemini_rubric.get('criteria', []))} criteria "
                    f"({'Groq' if gemini_rubric.get('groq_generated') else 'Mock'})")

        # Build rubric_criteria dict from generated rubric (used for comparison prompt)
        rubric_for_comparison = {
            c['name']: c['description']
            for c in gemini_rubric.get('criteria', [])
        } if gemini_rubric.get('criteria') else rubric_criteria

        # ── Step 2: Generate model answer from Gemini ────────────────────────
        logger.info("Generating model answer from Groq...")
        model_answer = self.groq_grader.generate_model_answer(questions, rubric_for_comparison)
        logger.info(f"Model answer generated: {len(model_answer)} characters")

        # ── Step 3: Run standard NLP pipeline ───────────────────────────────
        # BUG FIX: use the generated model_answer as the reference for all NLP
        # scoring — NOT the raw question text. Comparing an answer to a question
        # produces near-zero semantic similarity and 0 rubric coverage by design.
        reference_text = model_answer

        semantic_analysis     = self.semantic_comparator.calculate_semantic_similarity(student_text, reference_text)
        rubric_analysis       = self.rubric_engine.analyze_rubric_coverage(student_text, reference_text, rubric=gemini_rubric)
        grammar_analysis      = self.grammar_checker.analyze_grammar_and_style(student_text)
        factual_analysis      = self.fact_checker.verify_factual_accuracy(student_text, reference_text)
        completeness_analysis = self.qa_evaluator.analyze_completeness(student_text, reference_text)

        aggregated_result = self.scoring_aggregator.aggregate_scores(
            semantic_analysis, rubric_analysis, grammar_analysis,
            factual_analysis, completeness_analysis
        )

        # ── Step 4: Gemini comparison ────────────────────────────────────────
        groq_result = self.groq_grader.compare_with_model(
            student_text, model_answer, rubric_for_comparison
        )

        # ── Step 5: Combine scores ───────────────────────────────────────────
        # For handwritten/OCR scanned papers, Groq's LLM comparison is far more
        # reliable than NLP metrics (semantic/factual/rubric) which degrade badly
        # on OCR noise. Raise Groq weight to 0.70 so it dominates the final score.
        groq_weight    = 0.60 if self.use_groq else 0.10
        standard_weight = 1.0 - groq_weight

        # Normalise Groq score: LLMs tend to anchor low against their own verbose
        # model answers. Apply a calibration boost so that a "decent" score of 0.60
        # from Groq maps to ~75 on a 100-point scale, consistent with the rubric intent.
        raw_groq_score = groq_result.similarity_score * 100
        # Direct linear scale — no artificial floor.
        calibrated_groq_score = min(raw_groq_score, 100.0)

        enhanced_final_score = (
            aggregated_result['final_score'] * standard_weight +
            calibrated_groq_score * groq_weight
        )
        enhanced_confidence = (
            aggregated_result['confidence_score'] * standard_weight +
            groq_result.confidence * groq_weight
        )

        enhanced_feedback = {
            'strengths':   list(dict.fromkeys(
                aggregated_result['feedback']['strengths']  + groq_result.feedback['strengths'])),
            'weaknesses':  list(dict.fromkeys(
                aggregated_result['feedback']['weaknesses'] + groq_result.feedback['weaknesses'])),
            'suggestions': list(dict.fromkeys(
                aggregated_result['feedback']['suggestions']+ groq_result.feedback['suggestions'])),
        }

        processing_time = (datetime.now() - start_time).total_seconds()

        result = GradingResult(
            student_id=student_id,
            assignment_id=assignment_id,
            timestamp=datetime.now().isoformat(),
            semantic_score=semantic_analysis['score'],
            rubric_score=rubric_analysis['score'],
            grammar_score=grammar_analysis['score'],
            factual_score=factual_analysis['score'],
            completeness_score=completeness_analysis['score'],
            final_score=enhanced_final_score,
            word_count=grammar_analysis.get('vocabulary_analysis', {}).get('total_words', 0),
            sentence_count=grammar_analysis.get('sentence_analysis', {}).get('total_sentences', 0),
            technical_terms_found=len(semantic_analysis.get('statistics', {}).get('student_concepts', [])),
            grammar_issues=grammar_analysis.get('total_errors', 0),
            strengths=enhanced_feedback['strengths'],
            weaknesses=enhanced_feedback['weaknesses'],
            suggestions=enhanced_feedback['suggestions'],
            processing_time=processing_time,
            confidence_score=enhanced_confidence
        )

        # Attach all Groq-specific data
        result.gemini_model_answer    = model_answer
        result.gemini_rubric          = gemini_rubric
        result.gemini_similarity      = groq_result.similarity_score
        result.gemini_used            = groq_result.groq_used
        result.gemini_detailed_scores = groq_result.detailed_scores

        logger.info(f"Groq-enhanced grading completed for {student_id}: {result.final_score:.1f}/100")
        return result

    def get_system_info(self) -> Dict[str, Any]:
        return {
            'system_name': 'Advanced Grading System',
            'version': '2.1.0',
            'modules': [
                'Semantic Comparator', 'Grammar Checker', 'Rubric Engine',
                'Fact Checker', 'QA Evaluator', 'Scoring Aggregator', 'Groq AI Grader'
            ],
            'weights': self.scoring_aggregator.weights,
            'groq_enabled': self.use_groq,
            'groq_status': 'Active' if self.use_groq else 'Mock Mode',
        }