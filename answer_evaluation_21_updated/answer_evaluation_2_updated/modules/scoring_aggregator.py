import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ScoringAggregator:
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # Grammar and completeness get higher weight for handwritten/OCR papers
        # where semantic and rubric keyword matching is unreliable.
        self.weights = weights or {
            'semantic':     0.25,
            'rubric':       0.10,
            'grammar':      0.30,
            'factual':      0.15,
            'completeness': 0.20
        }
        
    def aggregate_scores(self, 
                        semantic_analysis: Dict[str, Any],
                        rubric_analysis: Dict[str, Any],
                        grammar_analysis: Dict[str, Any],
                        factual_analysis: Dict[str, Any],
                        completeness_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate all module scores into final result"""
        
        scores = {
            'semantic_score':     semantic_analysis.get('score', 0),
            'rubric_score':       rubric_analysis.get('score', 0),
            'grammar_score':      grammar_analysis.get('score', 0),
            'factual_score':      factual_analysis.get('score', 0),
            'completeness_score': completeness_analysis.get('score', 0)
        }
        
        final_score = (
            scores['semantic_score']     * self.weights['semantic'] +
            scores['rubric_score']       * self.weights['rubric'] +
            scores['grammar_score']      * self.weights['grammar'] +
            scores['factual_score']      * self.weights['factual'] +
            scores['completeness_score'] * self.weights['completeness']
        )
        
        confidence_score = self._calculate_confidence(
            semantic_analysis, rubric_analysis, grammar_analysis, 
            factual_analysis, completeness_analysis
        )
        
        feedback = self._generate_comprehensive_feedback(
            semantic_analysis, rubric_analysis, grammar_analysis, 
            factual_analysis, completeness_analysis
        )
        
        return {
            'final_score': min(final_score, 100.0),
            'individual_scores': scores,
            'confidence_score': confidence_score,
            'feedback': feedback,
            'weights_used': self.weights.copy(),
            'timestamp': datetime.now().isoformat()
        }
        
    def _calculate_confidence(self, semantic_analysis, rubric_analysis, grammar_analysis, 
                            factual_analysis, completeness_analysis) -> float:
        confidence_factors = []
        
        if semantic_analysis.get('phrase_overlap', 0) > 0.5:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
            
        if rubric_analysis.get('average_coverage', 0) > 0.7:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.7)
            
        error_density = grammar_analysis.get('error_density', 0)
        grammar_confidence = max(1.0 - error_density * 2, 0.3)
        confidence_factors.append(grammar_confidence)
        
        if factual_analysis.get('total_verifiable', 0) > 0:
            factual_confidence = factual_analysis.get('accuracy_ratio', 0)
        else:
            factual_confidence = 0.7
        confidence_factors.append(factual_confidence)
        
        completeness_confidence = min(completeness_analysis.get('content_coverage', 0) + 0.3, 1.0)
        confidence_factors.append(completeness_confidence)
        
        import numpy as np
        return np.mean(confidence_factors)
        
    def _generate_comprehensive_feedback(self, semantic_analysis, rubric_analysis, grammar_analysis, 
                                       factual_analysis, completeness_analysis) -> Dict[str, List[str]]:
        feedback = {
            'strengths': [],
            'weaknesses': [],
            'suggestions': []
        }
        
        if semantic_analysis.get('score', 0) > 80:
            feedback['strengths'].append("Strong conceptual understanding demonstrated")
        elif semantic_analysis.get('score', 0) < 60:
            feedback['weaknesses'].append("Conceptual understanding needs improvement")
            feedback['suggestions'].append("Review key concepts and their relationships")
            
        concepts_covered = rubric_analysis.get('concepts_covered', 0)
        total_concepts = rubric_analysis.get('total_concepts', 1)
        
        if concepts_covered >= total_concepts * 0.8:
            feedback['strengths'].append("Comprehensive coverage of key topics")
        elif concepts_covered < total_concepts * 0.5:
            feedback['weaknesses'].append("Missing coverage of several key concepts")
            feedback['suggestions'].append("Ensure all major topics are addressed")
            
        if grammar_analysis.get('score', 0) > 85:
            feedback['strengths'].append("Excellent grammar and writing quality")
        elif grammar_analysis.get('error_count', 0) > 5:
            feedback['weaknesses'].append("Multiple grammar and style issues")
            feedback['suggestions'].append("Proofread carefully and check grammar")
            
        if factual_analysis.get('contradicted_facts'):
            feedback['weaknesses'].append("Some factual inaccuracies detected")
            feedback['suggestions'].append("Verify technical facts and specifications")
        elif factual_analysis.get('verified_facts'):
            feedback['strengths'].append("Accurate technical information provided")
            
        if completeness_analysis.get('score', 0) > 80:
            feedback['strengths'].append("Comprehensive and complete response")
        elif completeness_analysis.get('length_ratio', 0) < 0.5:
            feedback['weaknesses'].append("Response appears incomplete or too brief")
            feedback['suggestions'].append("Expand your response with more detail and examples")
            
        return feedback
        
    def generate_detailed_report(self, aggregated_result: Dict[str, Any], 
                               student_id: str, assignment_id: str) -> Dict[str, Any]:
        report = {
            'student_id': student_id,
            'assignment_id': assignment_id,
            'timestamp': aggregated_result['timestamp'],
            'final_score': aggregated_result['final_score'],
            'confidence': aggregated_result['confidence_score'],
            'breakdown': {
                'semantic': {
                    'score': aggregated_result['individual_scores']['semantic_score'],
                    'weight': self.weights['semantic'],
                    'contribution': aggregated_result['individual_scores']['semantic_score'] * self.weights['semantic']
                },
                'rubric': {
                    'score': aggregated_result['individual_scores']['rubric_score'],
                    'weight': self.weights['rubric'],
                    'contribution': aggregated_result['individual_scores']['rubric_score'] * self.weights['rubric']
                },
                'grammar': {
                    'score': aggregated_result['individual_scores']['grammar_score'],
                    'weight': self.weights['grammar'],
                    'contribution': aggregated_result['individual_scores']['grammar_score'] * self.weights['grammar']
                },
                'factual': {
                    'score': aggregated_result['individual_scores']['factual_score'],
                    'weight': self.weights['factual'],
                    'contribution': aggregated_result['individual_scores']['factual_score'] * self.weights['factual']
                },
                'completeness': {
                    'score': aggregated_result['individual_scores']['completeness_score'],
                    'weight': self.weights['completeness'],
                    'contribution': aggregated_result['individual_scores']['completeness_score'] * self.weights['completeness']
                }
            },
            'feedback': aggregated_result['feedback'],
            'grade_level': self._determine_grade_level(aggregated_result['final_score']),
            'recommendations': self._generate_recommendations(aggregated_result)
        }
        return report
        
    def _determine_grade_level(self, score: float) -> str:
        if score >= 90:
            return "Excellent (A)"
        elif score >= 80:
            return "Good (B)"
        elif score >= 70:
            return "Satisfactory (C)"
        elif score >= 60:
            return "Needs Improvement (D)"
        else:
            return "Unsatisfactory (F)"
            
    def _generate_recommendations(self, aggregated_result: Dict[str, Any]) -> List[str]:
        recommendations = []
        scores = aggregated_result['individual_scores']
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        
        for component, score in sorted_scores[:2]:
            if score < 70:
                if component == 'semantic_score':
                    recommendations.append("Focus on understanding core concepts and their relationships")
                elif component == 'rubric_score':
                    recommendations.append("Ensure comprehensive coverage of all required topics")
                elif component == 'grammar_score':
                    recommendations.append("Improve writing quality and grammar accuracy")
                elif component == 'factual_score':
                    recommendations.append("Verify technical facts and specifications")
                elif component == 'completeness_score':
                    recommendations.append("Provide more detailed and comprehensive responses")
                    
        if not recommendations:
            recommendations.append("Continue maintaining high quality across all areas")
            
        return recommendations