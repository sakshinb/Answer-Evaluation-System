"""
PerQuestionRubric
-----------------
Manages per-question rubric definitions and weighted scoring.

Each question can have its own:
  * max_marks  – total marks allocated
  * criteria   – list of criterion objects (same schema as Groq-generated rubrics)
  * weights    – custom NLP-module weights (overrides global defaults)

The module handles:
  1. Storing / loading per-question rubric configs
  2. Computing weighted final scores per question
  3. Aggregating across questions using mark-weighted averaging
  4. Generating per-question feedback summaries

Schema (one question entry)
---------------------------
{
  "question_num": 1,
  "question_text": "Explain photosynthesis.",
  "max_marks": 10,
  "weights": {                          # optional — overrides globals
    "semantic": 0.30,
    "rubric":   0.25,
    "grammar":  0.10,
    "factual":  0.15,
    "completeness": 0.10
  },
  "criteria": [                         # optional Groq-style criteria
    {"name": "...", "description": "...", "max_marks": 4, "keywords": [...]}
  ]
}
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS = {
    "semantic":     0.30,
    "rubric":       0.25,
    "grammar":      0.20,
    "factual":      0.15,
    "completeness": 0.10,
}


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class PerQuestionRubric:
    """
    Stores and applies per-question rubric configurations.

    Usage
    -----
        pqr = PerQuestionRubric(question_rubrics=[...])
        score = pqr.compute_question_score(q_idx=0, scores_dict={...})
        total = pqr.aggregate_scores([score1, score2, ...])
    """

    def __init__(self, question_rubrics: Optional[List[Dict[str, Any]]] = None):
        """
        Parameters
        ----------
        question_rubrics : list of per-question dicts (see module docstring)
        """
        self._rubrics: List[Dict[str, Any]] = question_rubrics or []

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_rubrics(self, question_rubrics: List[Dict[str, Any]]):
        self._rubrics = question_rubrics

    def get_rubric(self, q_idx: int) -> Optional[Dict[str, Any]]:
        """Return the rubric for question index q_idx (0-based), or None."""
        if q_idx < len(self._rubrics):
            return self._rubrics[q_idx]
        return None

    def get_weights(self, q_idx: int) -> Dict[str, float]:
        """Return module weights for this question (falls back to defaults)."""
        rubric = self.get_rubric(q_idx)
        if rubric and rubric.get("weights"):
            w = rubric["weights"]
            # Normalise so they always sum to 1.0
            total = sum(w.values())
            if total > 0:
                return {k: v / total for k, v in w.items()}
        return _DEFAULT_WEIGHTS.copy()

    def get_max_marks(self, q_idx: int) -> float:
        """Return max marks for question q_idx, or 10 as default."""
        rubric = self.get_rubric(q_idx)
        if rubric:
            return float(rubric.get("max_marks", 10))
        return 10.0

    def get_criteria(self, q_idx: int) -> Optional[List[Dict]]:
        """Return the Groq-style criteria list for question q_idx, or None."""
        rubric = self.get_rubric(q_idx)
        if rubric:
            return rubric.get("criteria")
        return None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def compute_question_score(
        self,
        q_idx: int,
        component_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Apply per-question weights to the five component scores.

        Parameters
        ----------
        q_idx            : 0-based question index
        component_scores : dict with keys semantic, rubric, grammar, factual, completeness
                           (all on 0-100 scale)

        Returns
        -------
        dict with final_score (0-100), weighted breakdown, and max_marks
        """
        weights    = self.get_weights(q_idx)
        max_marks  = self.get_max_marks(q_idx)

        weighted_total = 0.0
        breakdown: Dict[str, Any] = {}

        for component, weight in weights.items():
            raw = component_scores.get(component + "_score",
                  component_scores.get(component, 0.0))
            contribution      = raw * weight
            weighted_total   += contribution
            breakdown[component] = {
                "raw_score":    round(raw, 2),
                "weight":       round(weight, 3),
                "contribution": round(contribution, 2),
            }

        final_pct   = min(weighted_total, 100.0)
        marks_earned = (final_pct / 100.0) * max_marks

        return {
            "question_index": q_idx,
            "final_score":    round(final_pct, 2),
            "marks_earned":   round(marks_earned, 2),
            "max_marks":      max_marks,
            "weights_used":   weights,
            "breakdown":      breakdown,
        }

    def aggregate_scores(self, per_question_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate per-question results into an overall score.
        Uses mark-weighted averaging (questions with more marks count more).

        Parameters
        ----------
        per_question_results : list of dicts from compute_question_score()

        Returns
        -------
        dict with overall_score (0-100), total_marks_earned, total_max_marks
        """
        if not per_question_results:
            return {"overall_score": 0.0, "total_marks_earned": 0.0, "total_max_marks": 0.0}

        total_earned = sum(r.get("marks_earned", 0.0) for r in per_question_results)
        total_max    = sum(r.get("max_marks", 10.0)   for r in per_question_results)
        overall_pct  = (total_earned / total_max * 100.0) if total_max > 0 else 0.0

        return {
            "overall_score":      round(overall_pct, 2),
            "total_marks_earned": round(total_earned, 2),
            "total_max_marks":    total_max,
            "per_question":       per_question_results,
        }

    # ------------------------------------------------------------------
    # Feedback helpers
    # ------------------------------------------------------------------

    def generate_score_card(self, aggregated: Dict[str, Any]) -> List[str]:
        """
        Return a list of human-readable score-card lines for the aggregated result.
        """
        lines = []
        total_max = aggregated.get("total_max_marks", 0)
        total_earned = aggregated.get("total_marks_earned", 0)
        lines.append(f"Overall: {total_earned:.1f} / {total_max:.0f} "
                     f"({aggregated.get('overall_score', 0):.1f}%)")

        for q_result in aggregated.get("per_question", []):
            q_idx    = q_result.get("question_index", "?")
            earned   = q_result.get("marks_earned", 0)
            max_m    = q_result.get("max_marks", 10)
            pct      = q_result.get("final_score", 0)
            lines.append(f"  Q{q_idx + 1}: {earned:.1f}/{max_m:.0f} ({pct:.1f}%)")

        return lines

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def from_ai_rubric(ai_rubric: Dict[str, Any],
                       questions: List[str],
                       equal_marks: float = 10.0) -> "PerQuestionRubric":
        """
        Build a PerQuestionRubric from a single AI-generated rubric
        by distributing the same criteria to every question.

        Parameters
        ----------
        ai_rubric    : dict from GroqGrader.generate_rubric()
        questions    : list of question strings
        equal_marks  : marks per question (default 10)
        """
        rubrics = []
        for i, q in enumerate(questions):
            rubrics.append({
                "question_num":  i + 1,
                "question_text": q,
                "max_marks":     equal_marks,
                "criteria":      ai_rubric.get("criteria", []),
            })
        return PerQuestionRubric(rubrics)

    @staticmethod
    def from_manual_config(config: List[Dict[str, Any]]) -> "PerQuestionRubric":
        """Build from a manually supplied list of per-question dicts."""
        return PerQuestionRubric(config)