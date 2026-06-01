"""
ListSetCoverage
---------------
Enhanced set-theoretic coverage scoring for list / enumeration questions.

Improvements over the basic _ListScorer:
* Synonym-aware matching via NLTK WordNet (optional)
* Partial credit for paraphrased items (fuzzy word overlap)
* Order-sensitivity penalty when reference is explicitly ordered (steps 1, 2, 3…)
* Extra-item bonus for correct additional valid items
* Negative marking option for clearly wrong items (off-topic with low overlap)
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _item_keywords(item: str, sw: set, lem) -> set:
    """Lemmatised content words from a single list item string."""
    from nltk.tokenize import word_tokenize
    return {lem.lemmatize(w.lower()) for w in word_tokenize(item)
            if w.isalpha() and w.lower() not in sw and len(w) > 2}


def _wordnet_synonyms(word: str) -> set:
    """Return a set of synset lemma names for word (lowercase)."""
    try:
        from nltk.corpus import wordnet
        syns = set()
        for ss in wordnet.synsets(word):
            for lemma in ss.lemmas():
                syns.add(lemma.name().lower().replace('_', ' '))
        return syns
    except Exception:
        return {word}


def _item_overlap(ref_kw: set, stud_kw: set, use_wordnet: bool = False) -> float:
    """
    Fraction of reference keywords covered by the student item.
    Optionally expands using WordNet synonyms.
    """
    if not ref_kw:
        return 0.0

    if use_wordnet:
        # Expand ref keywords with synonyms
        expanded = set()
        for w in ref_kw:
            expanded.update(_wordnet_synonyms(w))
        match = len(expanded & stud_kw) / len(ref_kw)
    else:
        match = len(ref_kw & stud_kw) / len(ref_kw)

    return min(match, 1.0)


def _is_ordered_list(text: str) -> bool:
    """Heuristic: returns True if the text uses explicit step/ordered numbering."""
    patterns = [
        r'^\s*\d+[\.)]\s',           # "1. " or "1) "
        r'\bstep\s+\d+\b',           # "Step 1"
        r'\bfirst(ly)?\b.*\bsecond(ly)?\b',  # first…second
    ]
    return any(re.search(p, text, re.I | re.M) for p in patterns)


def _extract_items(text: str) -> List[str]:
    """
    Split text into discrete list items.
    Tries bullet/numbered lines first, then sentence tokenization.
    """
    lines = text.strip().splitlines()
    items = [re.sub(r'^\s*[•\-\*\d]+[.)\s]+', '', l).strip() for l in lines]
    items = [i for i in items if len(i) > 3]
    if items:
        return [i.lower() for i in items]

    try:
        from nltk.tokenize import sent_tokenize
        return [s.lower() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        return [s.lower() for s in re.split(r'[.;]', text) if s.strip()]


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class ListSetCoverageScorer:
    """
    Scores list / enumeration answers using set-theoretic coverage.

    Parameters
    ----------
    use_wordnet     : bool  – expand synonyms via WordNet (slower but more robust)
    partial_threshold : float – minimum overlap to award partial credit (default 0.3)
    negative_marking  : bool  – penalise clearly off-topic extra items
    """

    def __init__(self,
                 use_wordnet: bool = True,
                 partial_threshold: float = 0.30,
                 negative_marking: bool = False):
        self.use_wordnet       = use_wordnet
        self.partial_threshold = partial_threshold
        self.negative_marking  = negative_marking

        try:
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            self._sw  = set(stopwords.words('english'))
            self._lem = WordNetLemmatizer()
        except Exception:
            self._sw  = set()
            self._lem = None

    # ------------------------------------------------------------------

    def score(self, student: str, reference: str) -> Dict[str, Any]:
        ref_items  = _extract_items(reference)
        stud_items = _extract_items(student)

        if not ref_items:
            return {"score": 50.0, "detail": {"method": "no_ref_items"}}

        ordered     = _is_ordered_list(reference)
        ref_kws     = [self._kw(i) for i in ref_items]
        stud_kws    = [self._kw(i) for i in stud_items]

        matched_scores: List[float] = []
        matched_indices: List[int]  = []   # stud item index used per ref item

        for r_idx, r_kw in enumerate(ref_kws):
            best_score = 0.0
            best_stud_idx = -1
            for s_idx, s_kw in enumerate(stud_kws):
                ov = _item_overlap(r_kw, s_kw, self.use_wordnet)
                if ov > best_score:
                    best_score    = ov
                    best_stud_idx = s_idx

            # Order penalty: if ordered list and student placed item far out of position
            if ordered and best_stud_idx >= 0:
                position_diff = abs(r_idx - best_stud_idx) / max(len(ref_items), 1)
                order_penalty = position_diff * 0.15          # up to 15% penalty
                best_score    = max(best_score - order_penalty, 0.0)

            if best_score >= self.partial_threshold:
                matched_scores.append(best_score)
                matched_indices.append(best_stud_idx)
            else:
                matched_scores.append(0.0)

        # Negative marking for clearly irrelevant extra items
        neg_penalty = 0.0
        if self.negative_marking and stud_items:
            used_stud_indices = set(matched_indices)
            extra_items = [s_kw for s_idx, s_kw in enumerate(stud_kws)
                           if s_idx not in used_stud_indices]
            for e_kw in extra_items:
                # If an extra item shares < 10% overlap with ANY ref item it's probably wrong
                max_ov = max((_item_overlap(r_kw, e_kw, False) for r_kw in ref_kws), default=0.0)
                if max_ov < 0.10:
                    neg_penalty += 0.02   # small deduction per wrong item

        # Coverage and extra-item bonus
        avg_match   = sum(matched_scores) / len(ref_items)
        n_matched   = sum(1 for s in matched_scores if s > 0)
        extra_bonus = min((len(stud_items) - len(ref_items)) * 0.015, 0.08) \
                      if len(stud_items) > len(ref_items) else 0.0

        raw_score  = min(avg_match + extra_bonus - neg_penalty, 1.0)
        final      = max(raw_score * 100, 0.0)

        return {
            "score": round(final, 2),
            "detail": {
                "method": "set_coverage" + ("_wordnet" if self.use_wordnet else ""),
                "ref_item_count":     len(ref_items),
                "student_item_count": len(stud_items),
                "items_matched":      n_matched,
                "avg_match_score":    round(avg_match, 3),
                "extra_bonus":        round(extra_bonus, 3),
                "neg_penalty":        round(neg_penalty, 3),
                "ordered_list":       ordered,
                "per_item_scores":    [round(s, 3) for s in matched_scores],
            }
        }

    def _kw(self, item: str) -> set:
        if self._lem is None:
            return {w.lower() for w in item.split() if len(w) > 2}
        return _item_keywords(item, self._sw, self._lem)