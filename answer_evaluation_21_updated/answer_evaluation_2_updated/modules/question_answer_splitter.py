"""
QuestionAnswerSplitter
----------------------
Aligns each section of a student's submission to its corresponding question
BEFORE any scoring happens.

Strategy (in order of confidence):
1. Explicit numbering  — "1.", "Q1.", "Question 1", "Ans 1:" etc.
2. Groq-based split    — ask the LLM to segment the text when no markers found
3. Fallback            — treat the entire submission as the answer to every question

Returns a list of (question, student_answer) pairs, one per question.
"""

import re
import json
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regex patterns for common student answer markers
# ---------------------------------------------------------------------------
_MARKER_PATTERNS = [
    # "Question 1", "Q1", "Q.1", "Ques 1"
    r"(?:question|ques|q)\.?\s*(\d+)\s*[:\-\.]?",
    # "Answer 1", "Ans 1", "A1", "A.1"
    r"(?:answer|ans|a)\.?\s*(\d+)\s*[:\-\.]?",
    # plain "1.", "1)", "(1)"
    r"^\s*\(?(\d+)[.)]\s",
]
_COMPILED = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _MARKER_PATTERNS]


def _find_numbered_segments(text: str, n_questions: int) -> Optional[List[str]]:
    """
    Try to split text into n_questions segments using explicit numbering markers.
    Returns a list of answer strings (one per question) or None if no markers found.
    """
    # Collect all marker positions and the question number they imply
    hits: List[Tuple[int, int]] = []  # (char_offset, question_number)
    for pattern in _COMPILED:
        for m in pattern.finditer(text):
            q_num = int(m.group(1))
            if 1 <= q_num <= n_questions:
                hits.append((m.start(), q_num))

    if not hits:
        return None

    # Sort by position, deduplicate by keeping the first hit for each question number
    hits.sort(key=lambda x: x[0])
    seen: dict = {}
    ordered: List[Tuple[int, int]] = []
    for offset, q_num in hits:
        if q_num not in seen:
            seen[q_num] = offset
            ordered.append((offset, q_num))

    # Need at least 2 markers to split meaningfully
    if len(ordered) < 2:
        return None

    segments: List[str] = [""] * n_questions
    for idx, (offset, q_num) in enumerate(ordered):
        start = offset
        end = ordered[idx + 1][0] if idx + 1 < len(ordered) else len(text)
        segments[q_num - 1] = text[start:end].strip()

    return segments if any(s for s in segments) else None


def _split_by_groq(client, model: str, text: str, questions: List[str]) -> Optional[List[str]]:
    """
    Ask the Groq LLM to identify which part of the submission answers each question.
    Returns a list of answer strings or None on failure.
    """
    questions_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    prompt = f"""You are a grading assistant. A student submitted one document answering multiple questions.
Your job is to EXTRACT the portion of text that answers each question.

QUESTIONS:
{questions_block}

STUDENT SUBMISSION:
{text[:4000]}

Return ONLY valid JSON in this exact format (no markdown, no extra text):
{{
  "splits": [
    {{"question_num": 1, "answer": "...extracted text for Q1..."}},
    {{"question_num": 2, "answer": "...extracted text for Q2..."}}
  ]
}}

Rules:
- Copy the student's words exactly, do not paraphrase or add content.
- If a question has no discernible answer in the text, set answer to "".
- Every question must appear in the splits list.
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        splits = data.get("splits", [])
        result = [""] * len(questions)
        for item in splits:
            idx = int(item["question_num"]) - 1
            if 0 <= idx < len(questions):
                result[idx] = item.get("answer", "").strip()
        return result
    except Exception as e:
        logger.warning("Groq-based splitting failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class QuestionAnswerSplitter:
    """
    Aligns a student's full submission text to individual questions.

    Usage:
        splitter = QuestionAnswerSplitter(groq_client, groq_model)
        pairs = splitter.split(student_text, questions)
        # pairs -> List[Tuple[str, str]]  [(question, student_answer), ...]
    """

    def __init__(self, groq_client=None, groq_model: str = "llama-3.3-70b-versatile"):
        self.client = groq_client
        self.model = groq_model

    def split(
        self,
        student_text: str,
        questions: List[str],
    ) -> List[Tuple[str, str]]:
        """
        Returns a list of (question, student_answer) tuples.
        The student_answer is the portion of text most likely answering that question.
        """
        n = len(questions)

        if n == 1:
            # Single-question submission — whole text is the answer
            return [(questions[0], student_text.strip())]

        # Strategy 1: regex-based numbered segments
        segments = _find_numbered_segments(student_text, n)
        if segments:
            logger.info("QuestionAnswerSplitter: used regex markers")
            return self._build_pairs(questions, segments, student_text)

        # Strategy 2: Groq-assisted split
        if self.client:
            segments = _split_by_groq(self.client, self.model, student_text, questions)
            if segments and any(s for s in segments):
                logger.info("QuestionAnswerSplitter: used Groq-based split")
                return self._build_pairs(questions, segments, student_text)

        # Strategy 3: fallback — map full text to every question
        logger.warning(
            "QuestionAnswerSplitter: could not split, using full text for all %d questions", n
        )
        return [(q, student_text.strip()) for q in questions]

    # ------------------------------------------------------------------
    def _build_pairs(
        self,
        questions: List[str],
        segments: List[str],
        full_text: str,
    ) -> List[Tuple[str, str]]:
        pairs = []
        for i, question in enumerate(questions):
            answer = segments[i] if i < len(segments) else ""
            # If a segment is empty, fall back to full text for that question
            pairs.append((question, answer.strip() if answer.strip() else full_text.strip()))
        return pairs