"""
Groq AI Grader - replaces Gemini for model answer generation and answer comparison.
Uses llama-3.3-70b-versatile via the Groq API.
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from groq import Groq

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"


@dataclass
class GroqResult:
    model_answer: str
    similarity_score: float
    detailed_scores: Dict[str, float]
    feedback: Dict[str, List[str]]
    confidence: float
    groq_used: bool


class GroqGrader:
    """Groq API integration — drop-in replacement for GeminiGrader."""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("GROQ_API_KEY", "")
        if not key:
            logger.warning("No GROQ_API_KEY found — falling back to mock mode.")
            self.use_mock = True
            self.client = None
        else:
            self.client = Groq(api_key=key)
            self.use_mock = False
            logger.info("GroqGrader initialised (model: %s)", GROQ_MODEL)

    # ------------------------------------------------------------------ #
    #  Public API (same signatures as GeminiGrader)                       #
    # ------------------------------------------------------------------ #

    def generate_model_answer(self, questions: List[str], rubric_criteria: Dict[str, str]) -> str:
        if self.use_mock:
            return self._mock_model_answer(questions)
        questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        criteria_text  = "\n".join(f"- {k}: {v}" for k, v in rubric_criteria.items())
        prompt = (
            "You are an expert academic evaluator. "
            "Write a concise but comprehensive model answer for a student exam. "
            "The answer should be proportionate to what a good student would write — "
            "aim for 200-400 words total, not an exhaustive textbook entry.\n\n"
            f"QUESTIONS:\n{questions_text}\n\n"
            f"EVALUATION CRITERIA:\n{criteria_text}\n\n"
            "Provide a well-structured answer that covers the key concepts clearly. "
            "Do NOT pad or over-explain — write what a strong student would write."
        )
        try:
            return self._chat(prompt)
        except Exception as e:
            logger.error("Groq model-answer generation failed: %s", e)
            return self._mock_model_answer(questions)

    def generate_rubric_from_gemini(self, questions: List[str], max_marks: int = 10) -> Dict[str, Any]:
        """Kept the same name so existing callers do not break."""
        return self.generate_rubric(questions, max_marks)

    def generate_rubric(self, questions: List[str], max_marks: int = 10) -> Dict[str, Any]:
        if self.use_mock:
            return self._mock_rubric(max_marks)
        questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        prompt = f"""You are an expert academic evaluator. Generate a grading rubric for:

QUESTIONS:
{questions_text}

TOTAL MARKS: {max_marks}

Return ONLY valid JSON (no markdown, no extra text):
{{
  "criteria": [
    {{"name": "Content Accuracy",       "description": "...", "max_marks": {round(max_marks*0.35)}, "keywords": [], "performance_levels": {{"excellent":"...","good":"...","average":"...","poor":"..."}}}},
    {{"name": "Completeness",           "description": "...", "max_marks": {round(max_marks*0.25)}, "keywords": [], "performance_levels": {{"excellent":"...","good":"...","average":"...","poor":"..."}}}},
    {{"name": "Clarity & Organization", "description": "...", "max_marks": {round(max_marks*0.20)}, "keywords": [], "performance_levels": {{"excellent":"...","good":"...","average":"...","poor":"..."}}}},
    {{"name": "Technical Terminology",  "description": "...", "max_marks": {round(max_marks*0.20)}, "keywords": [], "performance_levels": {{"excellent":"...","good":"...","average":"...","poor":"..."}}}}
  ],
  "total_marks": {max_marks},
  "general_instructions": "..."
}}"""
        try:
            raw = self._chat(prompt)
            return self._parse_rubric(raw, max_marks)
        except Exception as e:
            logger.error("Groq rubric generation failed: %s", e)
            return self._mock_rubric(max_marks)

    def compare_with_model(self, student_text: str, model_answer: str,
                           rubric_criteria: Dict[str, str]) -> GroqResult:
        if self.use_mock:
            return self._local_compare(student_text, model_answer)
        criteria_text = "\n".join(f"- {k}: {v}" for k, v in rubric_criteria.items())
        prompt = f"""You are a strict but fair academic examiner grading a student's assignment that was scanned and OCR-processed. OCR may introduce minor spelling noise — do NOT penalize for this.

Your task: score the student answer as a percentage (0–100) based on correctness and understanding compared to the reference answer.

REFERENCE ANSWER (what a complete answer covers):
{model_answer}

STUDENT ANSWER:
{student_text}

GRADING CRITERIA:
{criteria_text}

SCORING SCALE — apply strictly:
- 90–100: Covers nearly all key concepts correctly with clear understanding
- 75–89:  Covers most key concepts, minor gaps or small errors
- 60–74:  Covers several key concepts but has notable gaps or inaccuracies
- 40–59:  Partial understanding, significant gaps or errors
- 20–39:  Limited understanding, major concepts missing or wrong
- 0–19:   Fundamentally incorrect, off-topic, or essentially blank

RULES:
1. The reference answer is a complete ideal — students are NOT expected to match it word-for-word.
2. Score based on what the student DID cover correctly, not just what is missing.
3. Do NOT penalize for length alone — a concise correct answer can score 75+.
4. OCR noise (garbled words, odd spacing, symbol errors) must be ignored.
5. Be honest — if the answer is incomplete or has errors, reflect that in the score.
6. Do NOT inflate scores. A student covering ~50% of key points correctly should score 50–60.

Return ONLY valid JSON:
{{
  "similarity_score": 0.65,
  "detailed_scores": {{"content_accuracy": 0.70, "completeness": 0.55, "understanding": 0.65, "clarity": 0.60}},
  "strengths":   ["..."],
  "weaknesses":  ["..."],
  "suggestions": ["..."],
  "confidence":  0.85
}}

similarity_score must be between 0.0 and 1.0.
"""
        try:
            raw = self._chat(prompt)
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                d = json.loads(m.group())
                return GroqResult(
                    model_answer=model_answer,
                    similarity_score=float(d.get("similarity_score", 0.5)),
                    detailed_scores=d.get("detailed_scores", {}),
                    feedback={
                        "strengths":   d.get("strengths", []),
                        "weaknesses":  d.get("weaknesses", []),
                        "suggestions": d.get("suggestions", []),
                    },
                    confidence=float(d.get("confidence", 0.8)),
                    groq_used=True,
                )
        except Exception as e:
            logger.error("Groq comparison failed: %s", e)
        return self._local_compare(student_text, model_answer)

    def test_connection(self) -> Dict[str, Any]:
        if self.use_mock:
            return {"success": True, "message": "Mock mode (no API key)", "groq_available": False}
        try:
            reply = self._chat("Say: connection successful")
            return {"success": True, "message": "Groq connected", "groq_available": True,
                    "model": GROQ_MODEL, "response_preview": reply[:80]}
        except Exception as e:
            return {"success": False, "message": str(e), "groq_available": False}

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _chat(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2048,
        )
        return resp.choices[0].message.content.strip()

    def _chat_messages(self, messages: list) -> str:
        """Send a full messages array (with system prompt) to Groq."""
        resp = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=512,
        )
        return resp.choices[0].message.content.strip()

    def _parse_rubric(self, raw: str, max_marks: int) -> Dict[str, Any]:
        try:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                data["groq_generated"] = True
                data.setdefault("total_marks", max_marks)

                # Enforce correct marks distribution regardless of what the AI returned.
                # The AI often ignores the template values and assigns equal marks to all criteria.
                criteria = data.get("criteria", [])
                if criteria:
                    weights = [0.35, 0.25, 0.20, 0.20]
                    for i, criterion in enumerate(criteria):
                        w = weights[i] if i < len(weights) else (1.0 / len(criteria))
                        criterion["max_marks"] = max(1, round(max_marks * w))

                    # Adjust last criterion so total always sums to max_marks
                    assigned = sum(c["max_marks"] for c in criteria[:-1])
                    criteria[-1]["max_marks"] = max(1, max_marks - assigned)

                return data
        except Exception as e:
            logger.error("Rubric parse error: %s", e)
        return self._mock_rubric(max_marks)

    def _local_compare(self, student_text: str, model_answer: str) -> GroqResult:
        sw = set(student_text.lower().split())
        mw = set(model_answer.lower().split())
        overlap = len(sw & mw)
        union   = len(sw | mw)
        sim     = overlap / union if union else 0
        lr      = min(len(student_text) / max(len(model_answer), 1), 1.0)
        score   = sim * 0.7 + lr * 0.3
        return GroqResult(
            model_answer=model_answer,
            similarity_score=score,
            detailed_scores={"content_accuracy": sim, "completeness": lr},
            feedback={
                "strengths":   ["Good effort"] if score > 0.5 else [],
                "weaknesses":  ["Limited coverage"] if sim < 0.5 else [],
                "suggestions": ["Add more key concepts"] if sim < 0.5 else [],
            },
            confidence=0.6,
            groq_used=False,
        )

    def _mock_model_answer(self, questions: List[str]) -> str:
        return "\n\n".join(
            f"Question {i+1}: Comprehensive answer covering all key aspects of: {q}"
            for i, q in enumerate(questions)
        )

    def _mock_rubric(self, max_marks: int) -> Dict[str, Any]:
        # Distribute marks: 35% / 25% / 20% / remainder
        m1 = max(1, round(max_marks * 0.35))
        m2 = max(1, round(max_marks * 0.25))
        m3 = max(1, round(max_marks * 0.20))
        m4 = max(1, max_marks - m1 - m2 - m3)  # remainder so total always == max_marks
        return {
            "criteria": [
                {"name": "Content Accuracy",       "description": "Correctness of facts",        "max_marks": m1, "keywords": [], "performance_levels": {"excellent":"All correct","good":"Mostly correct","average":"Some correct","poor":"Mostly wrong"}},
                {"name": "Completeness",           "description": "Coverage of all topics",       "max_marks": m2, "keywords": [], "performance_levels": {"excellent":"Full coverage","good":"Most covered","average":"Partial","poor":"Minimal"}},
                {"name": "Clarity & Organization", "description": "Clear and logical structure",  "max_marks": m3, "keywords": [], "performance_levels": {"excellent":"Very clear","good":"Clear","average":"Somewhat clear","poor":"Unclear"}},
                {"name": "Technical Terminology",  "description": "Correct domain vocabulary",    "max_marks": m4, "keywords": [], "performance_levels": {"excellent":"Expert use","good":"Good use","average":"Limited","poor":"Absent"}},
            ],
            "total_marks": max_marks,
            "general_instructions": "Evaluate each criterion independently.",
            "groq_generated": False,
        }