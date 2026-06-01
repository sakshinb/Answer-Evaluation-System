import logging
from typing import Dict, List, Any, Set, Tuple
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


class SemanticComparator:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self._setup_nltk()

    def _setup_nltk(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def _extract_meaningful_words(self, text: str) -> Set[str]:
        words = set()
        for word in word_tokenize(text.lower()):
            if word.isalpha() and word not in self.stop_words and len(word) > 2:
                words.add(self.lemmatizer.lemmatize(word))
        return words

    def _extract_key_concepts(self, text: str) -> List[str]:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        key_concepts, current_phrase = [], []
        for word, tag in tagged:
            if (tag.startswith("NN") or tag.startswith("JJ")) and \
               word.lower() not in self.stop_words and word.isalpha():
                current_phrase.append(word.lower())
            else:
                if current_phrase:
                    key_concepts.append(" ".join(current_phrase))
                    current_phrase = []
        if current_phrase:
            key_concepts.append(" ".join(current_phrase))
        return [self.lemmatizer.lemmatize(c) for c in key_concepts if len(c) > 2]

    def _encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _max_pooled_similarity(self, query_embs: np.ndarray, corpus_embs: np.ndarray) -> np.ndarray:
        sims = query_embs @ corpus_embs.T
        return sims.max(axis=1)

    def calculate_document_similarity(self, student_text: str, reference_text: str) -> float:
        embs = self._encode([student_text, reference_text])
        return self._cosine(embs[0], embs[1])

    def calculate_sentence_similarity(self, student_text: str, reference_text: str) -> float:
        ref_sents = sent_tokenize(reference_text)
        stud_sents = sent_tokenize(student_text)
        if not ref_sents or not stud_sents:
            return 0.0
        ref_embs = self._encode(ref_sents)
        stud_embs = self._encode(stud_sents)
        # Use student-to-reference direction: each student sentence finds its best
        # matching reference sentence. This rewards coverage of what the student DID say
        # rather than penalising them for not covering every reference sentence.
        best_matches = self._max_pooled_similarity(stud_embs, ref_embs)
        return float(best_matches.mean())

    def calculate_content_coverage(self, student_text: str, reference_text: str,
                                   threshold: float = 0.55) -> Dict[str, Any]:
        ref_sents = sent_tokenize(reference_text)
        stud_sents = sent_tokenize(student_text)
        if not ref_sents:
            return {"average_coverage": 0.0, "minimum_coverage": 0.0,
                    "well_covered_sentences": 0, "total_reference_sentences": 0}
        if not stud_sents:
            return {"average_coverage": 0.0, "minimum_coverage": 0.0,
                    "well_covered_sentences": 0, "total_reference_sentences": len(ref_sents)}
        ref_embs = self._encode(ref_sents)
        stud_embs = self._encode(stud_sents)
        coverage_scores = self._max_pooled_similarity(ref_embs, stud_embs)
        return {
            "average_coverage": float(coverage_scores.mean()),
            "minimum_coverage": float(coverage_scores.min()),
            "well_covered_sentences": int((coverage_scores >= threshold).sum()),
            "total_reference_sentences": len(ref_sents),
        }

    def calculate_semantic_similarity(self, student_text: str, reference_text: str) -> Dict[str, Any]:
        """
        SBERT-based semantic similarity.

        Components:
          doc_sim   – whole-document cosine similarity           (weight 0.50)
          sent_sim  – mean best-match at sentence level          (weight 0.50)

        The coverage ratio is used only as a mild scale factor, not a heavy penalty,
        so concise but correct answers are not unfairly penalised.
        """
        doc_sim  = self.calculate_document_similarity(student_text, reference_text)
        sent_sim = self.calculate_sentence_similarity(student_text, reference_text)
        coverage = self.calculate_content_coverage(student_text, reference_text)

        coverage_ratio = (
            coverage["well_covered_sentences"] / coverage["total_reference_sentences"]
            if coverage["total_reference_sentences"] > 0 else 0.0
        )

        # Equal weighting between doc-level and sentence-level similarity.
        # sentence_similarity now measures how well student sentences match the reference
        # (student-to-reference direction), rewarding what was said correctly.
        raw_score = (doc_sim * 0.50 + sent_sim * 0.50) * 100.0

        # Very mild coverage dampener: max 5% reduction even with zero coverage
        coverage_factor = 0.95 + coverage["average_coverage"] * 0.05
        final_score = min(raw_score * coverage_factor, 100.0)

        student_words   = self._extract_meaningful_words(student_text)
        reference_words = self._extract_meaningful_words(reference_text)
        common_words    = student_words & reference_words

        return {
            "score": round(final_score, 2),
            "detailed_scores": {
                "document_similarity": round(doc_sim * 100, 2),
                "sentence_similarity": round(sent_sim * 100, 2),
                "coverage_score":      round(coverage_ratio * 100, 2),
            },
            "coverage_analysis": {
                "average_coverage":      round(coverage["average_coverage"] * 100, 2),
                "minimum_coverage":      round(coverage["minimum_coverage"] * 100, 2),
                "well_covered_sentences": coverage["well_covered_sentences"],
                "total_sentences":        coverage["total_reference_sentences"],
                "coverage_percentage":    round(coverage_ratio * 100, 2),
            },
            "statistics": {
                "common_words":          sorted(common_words),
                "total_reference_words": len(reference_words),
                "total_student_words":   len(student_words),
                "word_match_count":      len(common_words),
                "student_concepts":      self._extract_key_concepts(student_text),
                "reference_concepts":    self._extract_key_concepts(reference_text),
            },
        }