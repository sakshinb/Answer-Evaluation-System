#!/usr/bin/env python3


import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import requests
import re
import time

logger = logging.getLogger(__name__)

@dataclass
class GeminiResult:
    """Gemini grading result structure"""
    model_answer: str
    similarity_score: float
    detailed_scores: Dict[str, float]
    feedback: Dict[str, List[str]]
    confidence: float
    gemini_used: bool

class GeminiGrader:
    """Gemini API integration for the modular grading system"""
    
    # Try these models in order until one works
    MODELS_TO_TRY = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-001",
        "gemini-2.5-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash-lite-001",
    ]
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini grader"""
        HARDCODED_API_KEY = "AIzaSyCVe6Qx3pZWgx9JBljSAr31UnXBGhV0sbg"
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", HARDCODED_API_KEY)
        self.use_mock = False  # Always use real mode - API key is set
        self.base_url = None  # Will be set after finding working model
        self._working_model = None
        
        logger.info(f"Gemini grader initialized with real API key (mock mode: OFF)")
        # Find a working model at startup
        self._find_working_model()
    
    def _find_working_model(self):
        """Try each model until we find one that works with this API key"""
        for model in self.MODELS_TO_TRY:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            try:
                test_data = {
                    "contents": [{"parts": [{"text": "Hi"}]}],
                    "generationConfig": {"maxOutputTokens": 10}
                }

                # Try up to 3 times per model on 429
                for attempt in range(3):
                    r = requests.post(
                        f"{url}?key={self.api_key}",
                        headers={'Content-Type': 'application/json'},
                        json=test_data,
                        timeout=15
                    )
                    if r.status_code == 200:
                        self.base_url = url
                        self._working_model = model
                        self.use_mock = False
                        logger.info(f"✅ Working Gemini model found: {model}")
                        return
                    elif r.status_code == 429:
                        wait = 10 * (2 ** attempt)  # 10s, 20s, 40s
                        logger.warning(f"⏳ Model {model} rate limited. Waiting {wait}s (attempt {attempt+1}/3)…")
                        time.sleep(wait)
                        continue
                    else:
                        logger.warning(f"Model {model} returned {r.status_code}, trying next...")
                        break  # Non-429 error, move to next model

            except Exception as e:
                logger.warning(f"Model {model} failed: {e}, trying next...")

        # If we reach here, no model worked — switch to mock mode
        logger.error("❌ No working Gemini model found! Switching to mock mode.")
        logger.error("❌ Check your API key at: https://aistudio.google.com/app/apikey")
        self.use_mock = True
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    def generate_model_answer(self, questions: List[str], rubric_criteria: Dict[str, str]) -> str:
        """Generate model answer using Gemini API"""
        
        if self.use_mock:
            return self._generate_mock_model_answer(questions)
        
        try:
            prompt = self._create_model_answer_prompt(questions, rubric_criteria)
            response = self._call_gemini_api(prompt)
            answer = self._parse_model_answer_response(response)
            logger.info(f"✅ Gemini model answer generated ({len(answer)} chars)")
            return answer
        except Exception as e:
            logger.error(f"❌ Gemini model answer FAILED: {e}")
            return "\n\n".join([f"Question {i+1}: {q}\nAnswer: [Gemini API failed - {str(e)}]" for i, q in enumerate(questions)])

    def generate_rubric_from_gemini(self, questions: List[str], max_marks: int = 10) -> Dict[str, Any]:
        """
        Generate rubric criteria automatically from Gemini based on given questions.
        Returns a structured rubric dict with criteria, marks, and keywords.
        """
        if self.use_mock:
            return self._generate_mock_rubric(questions, max_marks)

        try:
            prompt = self._create_rubric_prompt(questions, max_marks)
            response = self._call_gemini_api(prompt)
            content = self._parse_model_answer_response(response)
            return self._parse_rubric_response(content, max_marks)
        except Exception as e:
            logger.error(f"Gemini rubric generation failed: {e}")
            return self._generate_mock_rubric(questions, max_marks)

    def _create_rubric_prompt(self, questions: List[str], max_marks: int) -> str:
        """Build the Gemini prompt for rubric generation."""
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        return f"""
You are an expert academic evaluator. Generate a detailed grading rubric for the following exam questions.
The rubric must help evaluate a student's written answer objectively.

QUESTIONS:
{questions_text}

TOTAL MARKS PER QUESTION: {max_marks}

Return ONLY a valid JSON object in the following format (no extra text, no markdown):
{{
  "criteria": [
    {{
      "name": "Content Accuracy",
      "description": "Correctness of facts and concepts",
      "max_marks": {round(max_marks * 0.35)},
      "keywords": ["keyword1", "keyword2"],
      "performance_levels": {{
        "excellent": "Description of excellent performance",
        "good": "Description of good performance",
        "average": "Description of average performance",
        "poor": "Description of poor performance"
      }}
    }},
    {{
      "name": "Completeness",
      "description": "Coverage of all required topics",
      "max_marks": {round(max_marks * 0.25)},
      "keywords": ["keyword1", "keyword2"],
      "performance_levels": {{
        "excellent": "...",
        "good": "...",
        "average": "...",
        "poor": "..."
      }}
    }},
    {{
      "name": "Clarity & Organization",
      "description": "Clear explanation and logical structure",
      "max_marks": {round(max_marks * 0.20)},
      "keywords": ["keyword1", "keyword2"],
      "performance_levels": {{
        "excellent": "...",
        "good": "...",
        "average": "...",
        "poor": "..."
      }}
    }},
    {{
      "name": "Technical Terminology",
      "description": "Use of correct domain-specific terms",
      "max_marks": {round(max_marks * 0.20)},
      "keywords": ["keyword1", "keyword2"],
      "performance_levels": {{
        "excellent": "...",
        "good": "...",
        "average": "...",
        "poor": "..."
      }}
    }}
  ],
  "total_marks": {max_marks},
  "general_instructions": "Brief note on how to apply this rubric"
}}
"""

    def _parse_rubric_response(self, content: str, max_marks: int) -> Dict[str, Any]:
        """Parse the Gemini rubric JSON response."""
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                rubric_data = json.loads(json_match.group())
                if 'criteria' not in rubric_data:
                    raise ValueError("Missing 'criteria' key in rubric response")
                rubric_data['gemini_generated'] = True
                rubric_data['total_marks'] = rubric_data.get('total_marks', max_marks)
                return rubric_data
        except Exception as e:
            logger.error(f"Failed to parse rubric JSON: {e}\nContent: {content[:300]}")
        return self._generate_mock_rubric([], max_marks)

    def _generate_mock_rubric(self, questions: List[str], max_marks: int = 10) -> Dict[str, Any]:
        """Generate a generic rubric when Gemini is unavailable."""
        return {
            "criteria": [
                {
                    "name": "Content Accuracy",
                    "description": "Correctness and relevance of the information presented",
                    "max_marks": round(max_marks * 0.35),
                    "keywords": ["correct", "accurate", "relevant", "fact"],
                    "performance_levels": {
                        "excellent": "All facts are correct and well-explained",
                        "good": "Most facts are correct with minor errors",
                        "average": "Some correct facts with notable gaps",
                        "poor": "Incorrect or missing essential information"
                    }
                },
                {
                    "name": "Completeness",
                    "description": "Coverage of all required aspects of the question",
                    "max_marks": round(max_marks * 0.25),
                    "keywords": ["complete", "comprehensive", "all", "covered"],
                    "performance_levels": {
                        "excellent": "All aspects comprehensively addressed",
                        "good": "Most aspects addressed",
                        "average": "Some aspects covered",
                        "poor": "Major aspects missing"
                    }
                },
                {
                    "name": "Clarity & Organization",
                    "description": "Logical structure and clear expression",
                    "max_marks": round(max_marks * 0.20),
                    "keywords": ["clear", "organized", "structured", "logical"],
                    "performance_levels": {
                        "excellent": "Excellent structure and clear language",
                        "good": "Good organization with minor clarity issues",
                        "average": "Some structure, parts are unclear",
                        "poor": "Disorganized and unclear"
                    }
                },
                {
                    "name": "Technical Terminology",
                    "description": "Correct use of domain-specific vocabulary",
                    "max_marks": round(max_marks * 0.20),
                    "keywords": ["terminology", "technical", "concept", "definition"],
                    "performance_levels": {
                        "excellent": "Consistently uses correct technical terms",
                        "good": "Uses most technical terms correctly",
                        "average": "Limited technical vocabulary",
                        "poor": "Incorrect or absent technical terms"
                    }
                }
            ],
            "total_marks": max_marks,
            "general_instructions": "Evaluate student answers against each criterion independently.",
            "gemini_generated": False
        }
    
    def _create_model_answer_prompt(self, questions: List[str], rubric_criteria: Dict[str, str]) -> str:
        """Create prompt for Gemini to generate model answer"""
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        criteria_text = "\n".join([f"- {k}: {v}" for k, v in rubric_criteria.items()])
        prompt = f"""
You are an expert academic evaluator. Generate comprehensive model answers for the following questions.

QUESTIONS:
{questions_text}

EVALUATION CRITERIA:
{criteria_text}

Please provide ideal model answers that:
1. Fully address each question
2. Meet all evaluation criteria
3. Demonstrate excellent understanding
4. Are comprehensive yet clear
5. Serve as benchmarks for student evaluation

Format your response as a clear, well-structured answer for each question.
"""
        return prompt
    
    def _call_gemini_api(self, prompt: str, retries: int = 3, backoff: float = 10.0) -> Dict[str, Any]:
        """Call Gemini API with the prompt — retries on 429 with exponential backoff"""
        headers = {'Content-Type': 'application/json'}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 2048,
            }
        }
        url = f"{self.base_url}?key={self.api_key}"

        for attempt in range(retries):
            response = requests.post(url, headers=headers, json=data, timeout=30)
            if response.status_code == 429:
                wait = backoff * (2 ** attempt)   # 10s, 20s, 40s
                logger.warning(f"⏳ Rate limited (429). Waiting {wait}s before retry {attempt+1}/{retries}…")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()

        # Final attempt after all retries exhausted
        response.raise_for_status()
        return response.json()
    
    def _parse_model_answer_response(self, response: Dict[str, Any]) -> str:
        """Parse Gemini API response"""
        try:
            content = response['candidates'][0]['content']['parts'][0]['text']
            return content.strip()
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing Gemini response: {e}")
            return "Error generating model answer from Gemini API"
    
    def _generate_mock_model_answer(self, questions: List[str]) -> str:
        """Generate mock model answer for demo purposes"""
        mock_answers = []
        for i, question in enumerate(questions):
            question_lower = question.lower()
            if any(word in question_lower for word in ['photosynthesis', 'plant', 'biology']):
                answer = "Photosynthesis is the process by which plants convert light energy into chemical energy..."
            elif any(word in question_lower for word in ['newton', 'motion', 'physics']):
                answer = "Newton's laws of motion: 1) Inertia 2) F=ma 3) Action-Reaction..."
            elif any(word in question_lower for word in ['operating', 'system', 'os', 'computer']):
                answer = "Operating systems manage computer resources including processes, memory, files..."
            else:
                answer = f"Model answer for: '{question}'. Covers fundamental principles with detailed explanations."
            mock_answers.append(f"Question {i+1}: {answer}")
        return "\n\n".join(mock_answers)
    
    def compare_with_model(self, student_text: str, model_answer: str, rubric_criteria: Dict[str, str]) -> GeminiResult:
        """Compare student answer with model answer using Gemini or local methods"""
        if not self.use_mock:
            try:
                # Small pause so back-to-back API calls don't hit rate limits
                time.sleep(5)
                result = self._gemini_comparison(student_text, model_answer, rubric_criteria)
                logger.info(f"✅ Gemini comparison completed (score: {result.similarity_score:.2f})")
                return result
            except Exception as e:
                logger.error(f"❌ Gemini comparison FAILED: {e}")
        logger.warning("⚠️ Using local comparison fallback")
        return self._local_comparison(student_text, model_answer)
    
    def _gemini_comparison(self, student_text: str, model_answer: str, rubric_criteria: Dict[str, str]) -> GeminiResult:
        """Use Gemini API for comparison"""
        criteria_text = "\n".join([f"- {k}: {v}" for k, v in rubric_criteria.items()])
        prompt = f"""
You are an expert academic evaluator. Compare the student's answer with the model answer and provide detailed scoring.

MODEL ANSWER (IDEAL):
{model_answer}

STUDENT ANSWER:
{student_text}

EVALUATION CRITERIA:
{criteria_text}

Please evaluate and provide scores as JSON:
{{
    "similarity_score": 0.85,
    "detailed_scores": {{
        "content_accuracy": 0.90,
        "completeness": 0.80,
        "understanding": 0.85,
        "clarity": 0.75
    }},
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "suggestions": ["suggestion1", "suggestion2"],
    "confidence": 0.90
}}

Provide numerical scores (0-1) and detailed feedback.
"""
        response = self._call_gemini_api(prompt)
        content = response['candidates'][0]['content']['parts'][0]['text']
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result_data = json.loads(json_match.group())
            return GeminiResult(
                model_answer=model_answer,
                similarity_score=result_data.get('similarity_score', 0.5),
                detailed_scores=result_data.get('detailed_scores', {}),
                feedback={
                    'strengths': result_data.get('strengths', []),
                    'weaknesses': result_data.get('weaknesses', []),
                    'suggestions': result_data.get('suggestions', [])
                },
                confidence=result_data.get('confidence', 0.8),
                gemini_used=True
            )
        return self._local_comparison(student_text, model_answer)
    
    def _local_comparison(self, student_text: str, model_answer: str) -> GeminiResult:
        """Local comparison as fallback"""
        student_words = set(student_text.lower().split())
        model_words = set(model_answer.lower().split())
        overlap = len(student_words & model_words)
        union = len(student_words | model_words)
        similarity = overlap / union if union > 0 else 0
        length_ratio = min(len(student_text) / len(model_answer), 1.0) if model_answer else 0
        overall_similarity = (similarity * 0.7 + length_ratio * 0.3)
        strengths, weaknesses, suggestions = [], [], []
        if overall_similarity > 0.8:
            strengths.append("Strong alignment with model answer")
        elif overall_similarity > 0.6:
            strengths.append("Good understanding demonstrated")
        if similarity < 0.5:
            weaknesses.append("Limited coverage of key concepts")
            suggestions.append("Include more key terms and concepts")
        if length_ratio < 0.5:
            weaknesses.append("Response appears incomplete")
            suggestions.append("Provide more detailed explanations")
        return GeminiResult(
            model_answer=model_answer,
            similarity_score=overall_similarity,
            detailed_scores={"content_accuracy": similarity, "completeness": length_ratio, "overall_quality": overall_similarity},
            feedback={'strengths': strengths, 'weaknesses': weaknesses, 'suggestions': suggestions},
            confidence=0.7,
            gemini_used=False
        )
    
    def enhance_grading_result(self, base_result: Dict[str, Any], questions: List[str], rubric_criteria: Dict[str, str]) -> Dict[str, Any]:
        """Enhance existing grading result with Gemini insights"""
        model_answer = self.generate_model_answer(questions, rubric_criteria)
        student_text = base_result.get('student_text', '')
        gemini_result = self.compare_with_model(student_text, model_answer, rubric_criteria)
        enhanced_result = base_result.copy()
        enhanced_result.update({
            'gemini_model_answer': gemini_result.model_answer,
            'gemini_similarity': gemini_result.similarity_score,
            'gemini_detailed_scores': gemini_result.detailed_scores,
            'gemini_feedback': gemini_result.feedback,
            'gemini_confidence': gemini_result.confidence,
            'gemini_used': gemini_result.gemini_used,
            'enhanced_final_score': (base_result.get('final_score', 0) + gemini_result.similarity_score * 100) / 2,
            'enhanced_confidence': (base_result.get('confidence_score', 0) + gemini_result.confidence) / 2
        })
        return enhanced_result
    
    def test_connection(self) -> Dict[str, Any]:
        """Test Gemini API connection"""
        if self.use_mock:
            return {'success': True, 'message': 'Running in mock mode', 'gemini_available': False}
        try:
            test_prompt = "Test connection. Respond with 'Connection successful.'"
            response = self._call_gemini_api(test_prompt)
            content = self._parse_model_answer_response(response)
            return {
                'success': True,
                'message': f'Gemini API connection successful (model: {self._working_model})',
                'gemini_available': True,
                'working_model': self._working_model,
                'response_preview': content[:100]
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Gemini API connection failed: {str(e)}',
                'gemini_available': False
            }