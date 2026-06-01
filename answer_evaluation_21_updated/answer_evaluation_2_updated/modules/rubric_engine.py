import logging
import math
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class RubricEngine:

    def _build_key_concepts_from_rubric(self, rubric: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Derive a key_concepts-style dict from a dynamic rubric so that
        get_missing_concepts / get_well_covered_concepts / generate_rubric_feedback
        all work without modification.

        Each rubric criterion becomes one concept key; its keywords list (or
        description words as a fallback) become the keyword list.
        Keywords are capped at 8 per criterion to avoid over-penalising students
        for not hitting every possible synonym.
        """
        key_concepts: Dict[str, List[str]] = {}
        for criterion in rubric.get("criteria", []):
            name = criterion.get("name", "Unknown")
            keywords = [kw.lower() for kw in criterion.get("keywords", [])]

            if not keywords:
                # Fall back to meaningful words from the description
                desc = criterion.get("description", "")
                keywords = [w.lower() for w in desc.split() if len(w) > 4]

            # Cap to 8 most important keywords to avoid diluting the ratio
            keywords = keywords[:8]

            # Use a stable dict key derived from the criterion name
            key = name.lower().replace(" ", "_")
            key_concepts[key] = keywords

        return key_concepts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_rubric_coverage(
        self,
        student_text: str,
        reference_text: str,
        rubric: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyse concept coverage.

        When *rubric* is supplied the criteria and keywords are taken from it
        directly (dynamic path).  Otherwise the method falls back to deriving
        rough signal from *reference_text* keyword frequency (legacy path).
        """
        if rubric and rubric.get("criteria"):
            return self._analyze_with_dynamic_rubric(student_text, rubric)
        return self._analyze_with_reference_text(student_text, reference_text)

    def get_missing_concepts(self, analysis: Dict[str, Any]) -> List[str]:
        """Return concepts whose coverage ratio is below 0.3."""
        return [
            concept.replace("_", " ").title()
            for concept, coverage in analysis["concept_coverage"].items()
            if coverage["coverage_ratio"] < 0.3
        ]

    def get_well_covered_concepts(self, analysis: Dict[str, Any]) -> List[str]:
        """Return concepts whose coverage ratio exceeds 0.7."""
        return [
            concept.replace("_", " ").title()
            for concept, coverage in analysis["concept_coverage"].items()
            if coverage["coverage_ratio"] > 0.7
        ]

    def generate_rubric_feedback(self, analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate strengths / weaknesses / suggestions from an analysis result."""
        feedback: Dict[str, List[str]] = {
            "strengths": [],
            "weaknesses": [],
            "suggestions": [],
        }

        well_covered = self.get_well_covered_concepts(analysis)
        missing_concepts = self.get_missing_concepts(analysis)

        if well_covered:
            feedback["strengths"].append(f"Strong coverage of: {', '.join(well_covered)}")

        if missing_concepts:
            feedback["weaknesses"].append(f"Limited coverage of: {', '.join(missing_concepts)}")
            feedback["suggestions"].append(f"Consider expanding on: {', '.join(missing_concepts)}")

        coverage_pct = analysis["average_coverage"] * 100
        if coverage_pct > 80:
            feedback["strengths"].append("Comprehensive topic coverage")
        elif coverage_pct < 50:
            feedback["suggestions"].append(
                "Include more key concepts and technical details"
            )

        return feedback

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_matches(kw_lower: str, student_text_lower: str, student_words: List[str]) -> bool:
        """
        Multi-strategy keyword matching:
        1. Exact substring (handles multi-word phrases)
        2. Stem match (first 5 chars) for plurals/conjugations
        3. Partial word match (keyword is prefix of a student word, min 4 chars)
        4. Abbreviation/acronym match (e.g. 'os' matches 'operating system' initials)
        """
        # 1. Exact substring
        if kw_lower in student_text_lower:
            return True

        # 2. Stem match (first 5 chars)
        if len(kw_lower) >= 5:
            stem = kw_lower[:5]
            if any(w.startswith(stem) for w in student_words):
                return True

        # 3. Partial prefix match (keyword >= 4 chars is prefix of any student word)
        if len(kw_lower) >= 4:
            if any(w.startswith(kw_lower) or kw_lower.startswith(w[:max(4, len(w)-1)])
                   for w in student_words if len(w) >= 4):
                return True

        # 4. Multi-word keyword: check if majority of its parts appear in student text
        kw_parts = kw_lower.split()
        if len(kw_parts) > 1:
            matched_parts = sum(1 for part in kw_parts if len(part) >= 3 and part in student_text_lower)
            if matched_parts >= max(1, len(kw_parts) - 1):  # allow 1 missing part
                return True

        return False

    def _compute_concept_coverage(
        self,
        student_text_lower: str,
        key_concepts: Dict[str, List[str]],
        max_marks_map: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Core keyword-matching loop shared by both analysis paths.

        *max_marks_map* is an optional dict of concept_key → max_marks drawn
        from the dynamic rubric.  When absent every concept is weighted equally.
        """
        concept_coverage: Dict[str, Any] = {}
        total_score = 0.0
        student_words = student_text_lower.split()

        for concept, keywords in key_concepts.items():
            if keywords:
                found_keywords = []
                for kw in keywords:
                    kw_lower = kw.lower().strip()
                    if self._keyword_matches(kw_lower, student_text_lower, student_words):
                        found_keywords.append(kw)
                found_count = len(found_keywords)
                raw_ratio = min(found_count / len(keywords), 1.0)
                # Square-root partial credit: hitting 50% of keywords gives ~70% score
                # instead of 50%, rewarding partial coverage more fairly.
                ratio = math.sqrt(raw_ratio)
            else:
                found_count = 0
                found_keywords = []
                ratio = 0.0

            entry: Dict[str, Any] = {
                "expected_keywords": len(keywords),
                "found_keywords": found_count,
                "coverage_ratio": raw_ratio if keywords else 0.0,  # raw ratio for threshold checks
                "coverage_score": ratio,                            # boosted ratio for scoring
                "keywords_found": found_keywords,
            }
            if max_marks_map:
                entry["max_marks"] = max_marks_map.get(concept, 1)

            concept_coverage[concept] = entry
            total_score += ratio

        return concept_coverage, total_score

    def _build_result(
        self,
        concept_coverage: Dict[str, Any],
        total_score: float,
        total_concepts: int,
    ) -> Dict[str, Any]:
        avg_coverage = total_score / total_concepts if total_concepts else 0.0
        return {
            "score": min(avg_coverage * 100, 100.0),
            "concept_coverage": concept_coverage,
            "average_coverage": avg_coverage,
            "concepts_covered": sum(
                1 for cov in concept_coverage.values() if cov["coverage_ratio"] > 0.5
            ),
            "total_concepts": total_concepts,
        }

    def _analyze_with_dynamic_rubric(
        self, student_text: str, rubric: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Dynamic path: concept list and keywords come entirely from the rubric."""
        student_text_lower = student_text.lower()
        key_concepts = self._build_key_concepts_from_rubric(rubric)

        max_marks_map = {
            criterion.get("name", "Unknown").lower().replace(" ", "_"): criterion.get(
                "max_marks", 1
            )
            for criterion in rubric.get("criteria", [])
        }

        concept_coverage, total_score = self._compute_concept_coverage(
            student_text_lower, key_concepts, max_marks_map
        )
        return self._build_result(concept_coverage, total_score, len(key_concepts))

    def _analyze_with_reference_text(
        self, student_text: str, reference_text: str
    ) -> Dict[str, Any]:
        """
        Legacy fallback path: infer key concepts by comparing keyword frequencies
        between the student answer and the reference answer.

        The reference text is scanned for informative words (length > 4) and
        grouped into a single synthetic concept so the rest of the pipeline
        keeps working.
        """
        student_text_lower = student_text.lower()
        reference_text_lower = reference_text.lower()

        ref_words = list(
            {w for w in reference_text_lower.split() if len(w) > 4}
        )

        key_concepts = {"reference_content": ref_words}
        concept_coverage, total_score = self._compute_concept_coverage(
            student_text_lower, key_concepts
        )

        # Also track per-word detail for richer feedback
        found_words = concept_coverage["reference_content"]["keywords_found"]
        concept_coverage["reference_content"]["detail"] = (
            f"{len(found_words)} of {len(ref_words)} reference terms found"
        )

        return self._build_result(concept_coverage, total_score, len(key_concepts))