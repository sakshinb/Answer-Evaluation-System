"""
AdaptiveWeightEngine
--------------------
Dynamically adjusts NLP-module weights based on:

1. Question type   (grammar weight ↓ for code/numerical; semantic ↑ for conceptual)
2. Student profile (grammar weight ↓ for non-native speakers; semantic ↑ for code Qs)
3. Answer length   (completeness weight ↑ for very short answers)
4. Detected language difficulty

The engine returns a normalised weight dict compatible with ScoringAggregator.

Usage
-----
    engine = AdaptiveWeightEngine()
    weights = engine.compute_weights(
        question_type   = "code",        # from QuestionTypeClassifier
        is_native       = False,         # detected / declared
        answer_length   = 120,           # word count
        base_weights    = {...}          # optional starting point
    )
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Base weights (same as system default)
_BASE = {
    "semantic":     0.30,
    "rubric":       0.25,
    "grammar":      0.20,
    "factual":      0.15,
    "completeness": 0.10,
}

# Per-type overrides (deltas added to base, then renormalised)
_TYPE_DELTAS: Dict[str, Dict[str, float]] = {
    "code": {
        "semantic":     +0.10,   # AST/code similarity most important
        "grammar":      -0.15,   # grammar barely matters in code
        "rubric":       +0.05,
        "completeness": +0.00,
        "factual":      +0.00,
    },
    "numerical": {
        "semantic":     +0.15,   # numerical accuracy
        "grammar":      -0.15,
        "rubric":       +0.05,
        "factual":      -0.05,
        "completeness": +0.00,
    },
    "factual": {
        "factual":      +0.10,   # accuracy most critical
        "semantic":     +0.05,
        "grammar":      -0.05,
        "rubric":       -0.05,
        "completeness": -0.05,
    },
    "analytical": {
        "semantic":     +0.05,
        "rubric":       +0.05,
        "grammar":      +0.05,
        "factual":      -0.05,
        "completeness": -0.10,
    },
    "list": {
        "rubric":       +0.10,   # set coverage
        "semantic":     -0.05,
        "grammar":      -0.05,
        "factual":      +0.00,
        "completeness": +0.00,
    },
    "conceptual": {
        # balanced — keep base
    },
}

# Non-native-speaker adjustment (grammar weight → semantic weight)
_NON_NATIVE_GRAMMAR_REDUCTION = 0.08   # reduce grammar by this much
_NON_NATIVE_SEMANTIC_BOOST    = 0.05   # add to semantic
_NON_NATIVE_RUBRIC_BOOST      = 0.03   # add to rubric


def _normalise(weights: Dict[str, float]) -> Dict[str, float]:
    """Ensure weights sum to 1.0 and are non-negative."""
    clipped = {k: max(v, 0.02) for k, v in weights.items()}   # floor at 2%
    total   = sum(clipped.values())
    return {k: round(v / total, 4) for k, v in clipped.items()}


class AdaptiveWeightEngine:
    """
    Computes adaptive NLP-module weights.

    Parameters
    ----------
    base_weights : optional starting weights dict (defaults to system defaults)
    """

    def __init__(self, base_weights: Optional[Dict[str, float]] = None):
        self._base = (base_weights or _BASE).copy()

    def compute_weights(
        self,
        question_type:  str  = "conceptual",
        is_native:      bool = True,
        answer_length:  int  = 100,
        answer_has_code: bool = False,
        base_weights:   Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Return a normalised weight dict.

        Parameters
        ----------
        question_type   : one of factual/conceptual/analytical/numerical/code/list
        is_native       : True if student is a native English speaker
        answer_length   : word count of the student answer
        answer_has_code : True if the answer contains code blocks
        base_weights    : override the engine's default base (per-session config)
        """
        weights = (base_weights or self._base).copy()

        # ── 1. Apply question-type deltas ──────────────────────────────
        deltas = _TYPE_DELTAS.get(question_type, {})
        for k, delta in deltas.items():
            weights[k] = weights.get(k, 0.0) + delta

        # ── 2. Code-in-answer override (regardless of classified type) ──
        if answer_has_code and question_type not in ("code", "numerical"):
            weights["grammar"]  = max(weights.get("grammar", 0.20) - 0.08, 0.05)
            weights["semantic"] = weights.get("semantic", 0.30) + 0.08

        # ── 3. Non-native speaker adjustment ──────────────────────────
        if not is_native:
            # Grammar mistakes are expected → reduce its weight
            g = weights.get("grammar", 0.20)
            reduction = min(g - 0.05, _NON_NATIVE_GRAMMAR_REDUCTION)  # don't drop below 5%
            weights["grammar"]  = g - reduction
            weights["semantic"] = weights.get("semantic", 0.30) + _NON_NATIVE_SEMANTIC_BOOST
            weights["rubric"]   = weights.get("rubric",   0.25) + _NON_NATIVE_RUBRIC_BOOST

        # ── 4. Short-answer penalty: boost completeness weight ─────────
        if answer_length < 50:
            weights["completeness"] = weights.get("completeness", 0.10) + 0.08
            weights["semantic"]     = max(weights.get("semantic", 0.30) - 0.04, 0.10)

        # ── 5. Normalise ───────────────────────────────────────────────
        return _normalise(weights)

    def explain(self, weights: Dict[str, float], question_type: str,
                is_native: bool, answer_length: int) -> Dict[str, str]:
        """Return human-readable explanation for each weight choice."""
        notes = {}
        deltas = _TYPE_DELTAS.get(question_type, {})

        for component, w in weights.items():
            reasons = []
            if component in deltas and deltas[component] != 0:
                direction = "increased" if deltas[component] > 0 else "decreased"
                reasons.append(f"{direction} for '{question_type}' questions")
            if component == "grammar" and not is_native:
                reasons.append("reduced: non-native speaker detected")
            if component == "completeness" and answer_length < 50:
                reasons.append("boosted: very short answer")
            notes[component] = f"{w:.1%}" + (f" ({'; '.join(reasons)})" if reasons else "")
        return notes

    def detect_native_speaker(self, student_text: str) -> bool:
        """
        Lightweight heuristic: estimates native English speaker status
        from lexical diversity, sentence length distribution, and
        frequency of common ESL error patterns.

        Returns True (likely native) or False (possibly non-native).
        This is advisory only — teachers can always override.
        """
        if not student_text or len(student_text) < 50:
            return True   # not enough evidence

        words = student_text.lower().split()
        if not words:
            return True

        # ESL error patterns common in non-native writing
        esl_patterns = [
            r'\bthe\s+\w+\s+is\s+very\s+(much|very)\b',   # double intensifier
            r'\bmake\s+a\s+(?:research|progress|homework)\b',  # wrong article
            r'\b(?:informations|advices|equipments|furnitures)\b',  # wrong plural
            r'\b(?:i am agree|i am disagree)\b',
            r'\bdiscuss about\b',
            r'\bexplain about\b',
        ]
        import re
        esl_hits = sum(1 for p in esl_patterns if re.search(p, student_text, re.I))

        # Lexical diversity (low TTR can indicate non-native)
        unique_words  = len(set(words))
        ttr           = unique_words / len(words)

        # Average word length (non-native often use shorter, simpler words)
        avg_len = sum(len(w) for w in words) / len(words)

        # Scoring: more ESL hits or very low diversity → likely non-native
        score = 0
        if esl_hits >= 1: score += 2
        if ttr < 0.25:    score += 1
        if avg_len < 3.0: score += 1
        
        return score < 2   # < 2 → treat as native