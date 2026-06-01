"""
QuestionTypeClassifier
----------------------
Detects the type of each question and routes it to the correct specialised scorer.

Question types supported:
  factual     — "What is X?", "Who invented Y?", "When did Z happen?"
  conceptual  — "Explain X", "Describe how Y works", "What is the difference between..."
  analytical  — "Compare X and Y", "Evaluate the impact of...", "Critically analyse..."
  numerical   — "Calculate...", "Solve for...", "Derive the formula...", equations with =
  code        — "Write a program...", "Implement...", code snippets in the answer
  list        — "List three...", "Name five...", "Give examples of..."

Architecture
------------
1. QuestionTypeClassifier.classify(question)  -> str (one of the 6 type constants)
2. SpecialisedScorer.score(student, reference) -> Dict with 'score' (0-100) and 'detail'
3. RoutedScorer.score(question, student, reference) -> RoutedScore dataclass
"""

import re
import ast
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Question type constants ────────────────────────────────────────────────────
FACTUAL     = "factual"
CONCEPTUAL  = "conceptual"
ANALYTICAL  = "analytical"
NUMERICAL   = "numerical"
CODE        = "code"
LIST_TYPE   = "list"


# ── Keyword trigger lists ──────────────────────────────────────────────────────
_FACTUAL_TRIGGERS = [
    r"\bwho\b", r"\bwhen\b", r"\bwhere\b", r"\bwhich\b",
    r"\bwhat is the (name|date|year|capital|author|inventor|founder)\b",
    r"\bstate the\b", r"\bidentify\b", r"\bdefine\b",
]

_CONCEPTUAL_TRIGGERS = [
    r"\bexplain\b", r"\bdescribe\b", r"\bwhat is\b", r"\bhow does\b",
    r"\bwhat are\b", r"\bwhy\b", r"\belaborate\b", r"\bdiscuss\b",
    r"\billustrate\b", r"\bwhat do you (mean|understand)\b",
    r"\bdifference between\b", r"\bdistinguish\b",
]

_ANALYTICAL_TRIGGERS = [
    r"\bcompare\b", r"\bcontrast\b", r"\banalyse\b", r"\banalyze\b",
    r"\bevaluate\b", r"\bcritically\b", r"\bjustify\b", r"\bassess\b",
    r"\bimpact of\b", r"\badvantages? and disadvantages?\b",
    r"\bpros? and cons?\b", r"\bto what extent\b", r"\binterpret\b",
]

_NUMERICAL_TRIGGERS = [
    r"\bcalculate\b", r"\bcompute\b", r"\bsolve\b",
    r"\bfind the (value|area|volume|mass|speed|force)\b",
    r"\bderive\b", r"\bprove\b", r"\bshow that\b",
    r"\bwhat is the (value|result|answer) of\b",
    r"[=+\-*/^]\s*\d", r"\d\s*[=+\-*/^]",
]

_CODE_TRIGGERS = [
    r"\bwrite a (program|function|script|code|class|method|algorithm)\b",
    r"\bimplement\b", r"\bcode\b", r"\bprogram\b", r"\balgorithm\b",
    r"\bpseudocode\b", r"\bdebug\b",
    r"\boutput of the (following|given) (program|code)\b",
    r"```", r"\bdef \w+\(", r"\bclass \w+",
]

_LIST_TRIGGERS = [
    r"\blist\b", r"\bname\b", r"\bgive\b.*\b(example|instance|type|kind)\b",
    r"\benumerate\b", r"\bmention\b", r"\bstate\s+(any\s+)?\d",
    r"\b\d+\s+(types?|kinds?|examples?|reasons?|ways?|steps?|features?|advantages?|disadvantages?)\b",
    r"\bwhat are the (types?|kinds?|examples?|reasons?|steps?|features?)\b",
]


def _matches_any(text: str, patterns: List[str]) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


# ── Classifier ─────────────────────────────────────────────────────────────────

class QuestionTypeClassifier:
    """
    Classify a question into one of 6 types.

    Priority order (highest score wins; priority breaks ties):
        code > numerical > list > analytical > factual > conceptual
    """

    def classify(self, question: str) -> str:
        q = question.strip()
        scores: Dict[str, int] = {
            CODE: 0, NUMERICAL: 0, LIST_TYPE: 0,
            ANALYTICAL: 0, FACTUAL: 0, CONCEPTUAL: 0,
        }

        if _matches_any(q, _CODE_TRIGGERS):      scores[CODE]      += 3
        if _matches_any(q, _NUMERICAL_TRIGGERS): scores[NUMERICAL] += 3
        if _matches_any(q, _LIST_TRIGGERS):      scores[LIST_TYPE] += 3
        if _matches_any(q, _ANALYTICAL_TRIGGERS):scores[ANALYTICAL]+= 2
        if _matches_any(q, _FACTUAL_TRIGGERS):   scores[FACTUAL]   += 2
        if _matches_any(q, _CONCEPTUAL_TRIGGERS):scores[CONCEPTUAL]+= 1

        priority = [CODE, NUMERICAL, LIST_TYPE, ANALYTICAL, FACTUAL, CONCEPTUAL]
        best = max(priority, key=lambda t: scores[t])
        if scores[best] == 0:
            best = CONCEPTUAL

        logger.info("QuestionTypeClassifier: '%s...' -> %s", q[:60], best)
        return best

    def classify_all(self, questions: List[str]) -> List[str]:
        return [self.classify(q) for q in questions]


# ── Specialised scorers ────────────────────────────────────────────────────────

class _FactualScorer:
    """Entity + number overlap; contradiction penalty for wrong numbers."""

    def score(self, student: str, reference: str) -> Dict[str, Any]:
        from nltk.tokenize import word_tokenize
        from nltk import pos_tag

        def entities(text):
            tokens = word_tokenize(text)
            tagged = pos_tag(tokens)
            ents = {w.lower() for w, t in tagged if t.startswith("NNP")}
            nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", text))
            return ents | nums

        ref_ents  = entities(reference)
        stud_ents = entities(student)

        if not ref_ents:
            from nltk.corpus import stopwords
            sw = set(stopwords.words("english"))
            rw = {w.lower() for w in word_tokenize(reference) if w.isalpha() and w.lower() not in sw}
            sw2 = {w.lower() for w in word_tokenize(student) if w.isalpha() and w.lower() not in sw}
            ov = len(rw & sw2) / len(rw) if rw else 0.0
            return {"score": round(ov * 100, 2), "detail": {"method": "word_overlap"}}

        correct = ref_ents & stud_ents
        missing = ref_ents - stud_ents
        accuracy = len(correct) / len(ref_ents)

        contradiction_penalty = 0.0
        ref_nums  = set(re.findall(r"\b\d+(?:\.\d+)?\b", reference))
        stud_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", student))
        if ref_nums and stud_nums and not (ref_nums & stud_nums):
            contradiction_penalty = 0.20

        score = max(accuracy - contradiction_penalty, 0.0) * 100
        return {
            "score": round(score, 2),
            "detail": {
                "correct_entities": len(correct),
                "missing_entities": len(missing),
                "total_ref_entities": len(ref_ents),
                "contradiction_penalty": contradiction_penalty > 0,
            }
        }


class _ConceptualScorer:
    """Concept coverage + explanation depth + examples/definitions/transitions."""

    _EXAMPLE_MARKERS    = re.compile(r"\bfor example\b|\bfor instance\b|\bsuch as\b|\be\.g\b|\blike\b|\bconsider\b", re.I)
    _DEFINITION_MARKERS = re.compile(r"\bis defined as\b|\brefers to\b|\bmeans\b|\bis called\b|\bknown as\b", re.I)
    _TRANSITION_WORDS   = re.compile(r"\bhowever\b|\btherefore\b|\bfurthermore\b|\bmoreover\b|\bconsequently\b|\badditionally\b|\bin contrast\b|\bon the other hand\b", re.I)

    def score(self, student: str, reference: str) -> Dict[str, Any]:
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        sw  = set(stopwords.words("english"))
        lem = WordNetLemmatizer()

        def kw(text):
            return {lem.lemmatize(w.lower()) for w in word_tokenize(text)
                    if w.isalpha() and w.lower() not in sw and len(w) > 2}

        ref_kw  = kw(reference)
        stud_kw = kw(student)
        concept_coverage = len(ref_kw & stud_kw) / len(ref_kw) if ref_kw else 0.0
        raw_depth = len(sent_tokenize(student)) / max(len(sent_tokenize(reference)), 1)
        depth = min(raw_depth, 1.5) / 1.5
        has_ex  = 1.0 if self._EXAMPLE_MARKERS.search(student)    else 0.0
        has_def = 1.0 if self._DEFINITION_MARKERS.search(student) else 0.0
        has_tr  = 1.0 if self._TRANSITION_WORDS.search(student)   else 0.0

        score = (concept_coverage * 0.45 + depth * 0.20 + has_ex * 0.15 + has_def * 0.10 + has_tr * 0.10) * 100
        return {
            "score": round(min(score, 100.0), 2),
            "detail": {
                "concept_coverage_pct": round(concept_coverage * 100, 2),
                "depth_pct": round(depth * 100, 2),
                "has_examples": bool(has_ex),
                "has_definitions": bool(has_def),
                "has_transitions": bool(has_tr),
            }
        }


class _AnalyticalScorer:
    """Rewards structured argument, counterpoints, evidence, and coverage."""

    _ARGUMENT_MARKERS = re.compile(r"\bbecause\b|\bsince\b|\bdue to\b|\bas a result\b|\bthis shows\b|\bthus\b|\bhence\b|\bleads to\b", re.I)
    _COUNTER_MARKERS  = re.compile(r"\bhowever\b|\bon the other hand\b|\bin contrast\b|\bconversely\b|\balthough\b|\bdespite\b|\bnevertheless\b", re.I)
    _EVIDENCE_MARKERS = re.compile(r"\baccording to\b|\bdata (shows?|suggests?)\b|\bresearch\b|\bstudies? (show|indicate|suggest)\b|\bstatistically\b", re.I)

    def score(self, student: str, reference: str) -> Dict[str, Any]:
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        sw  = set(stopwords.words("english"))
        lem = WordNetLemmatizer()

        def kw(text):
            return {lem.lemmatize(w.lower()) for w in word_tokenize(text)
                    if w.isalpha() and w.lower() not in sw and len(w) > 2}

        ref_kw = kw(reference)
        kw_cov = len(ref_kw & kw(student)) / len(ref_kw) if ref_kw else 0.0
        depth  = min(len(sent_tokenize(student)) / max(len(sent_tokenize(reference)), 1), 1.0)
        arg    = 1.0 if self._ARGUMENT_MARKERS.search(student) else 0.0
        ctr    = 1.0 if self._COUNTER_MARKERS.search(student)  else 0.0
        evid   = 1.0 if self._EVIDENCE_MARKERS.search(student) else 0.0

        score = (kw_cov * 0.35 + depth * 0.20 + arg * 0.20 + ctr * 0.15 + evid * 0.10) * 100
        return {
            "score": round(min(score, 100.0), 2),
            "detail": {
                "kw_coverage_pct": round(kw_cov * 100, 2),
                "depth_pct": round(depth * 100, 2),
                "has_arguments": bool(arg),
                "has_counterpoints": bool(ctr),
                "has_evidence": bool(evid),
            }
        }


class _NumericalScorer:
    """Numeric value comparison with optional sympy expression equivalence."""

    def _nums(self, text: str) -> List[float]:
        results = []
        for r in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text):
            try:
                results.append(float(r))
            except ValueError:
                pass
        return results

    def _sympy_check(self, student: str, reference: str) -> Optional[float]:
        try:
            import sympy
            def last_expr(text):
                m = re.findall(r"=\s*([-\d\.\+\-\*/\(\)\^\s]+)", text)
                return m[-1].strip() if m else None
            re_expr  = last_expr(reference)
            st_expr  = last_expr(student)
            if re_expr and st_expr:
                rv = float(sympy.sympify(re_expr.replace("^", "**")))
                sv = float(sympy.sympify(st_expr.replace("^", "**")))
                if abs(rv) > 1e-10:
                    return max(1.0 - abs(rv - sv) / abs(rv), 0.0)
                return 1.0 if abs(sv - rv) < 1e-6 else 0.0
        except Exception:
            pass
        return None

    def score(self, student: str, reference: str) -> Dict[str, Any]:
        sp = self._sympy_check(student, reference)
        if sp is not None:
            return {"score": round(sp * 100, 2), "detail": {"method": "sympy", "match": sp}}

        ref_nums  = self._nums(reference)
        stud_nums = self._nums(student)

        if not ref_nums:
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            sw = set(stopwords.words("english"))
            rw  = {w.lower() for w in word_tokenize(reference) if w.isalpha() and w.lower() not in sw}
            sw2 = {w.lower() for w in word_tokenize(student) if w.isalpha() and w.lower() not in sw}
            ov = len(rw & sw2) / len(rw) if rw else 0.0
            return {"score": round(ov * 100, 2), "detail": {"method": "word_fallback"}}

        matched = 0
        for rn in ref_nums:
            for sn in stud_nums:
                tol = abs(rn) * 0.01 if abs(rn) > 1e-10 else 1e-6
                if abs(rn - sn) <= tol:
                    matched += 1
                    break

        ratio = matched / len(ref_nums)
        final_bonus = 0.0
        if ref_nums and stud_nums:
            rn, sn = ref_nums[-1], stud_nums[-1]
            tol = abs(rn) * 0.01 if abs(rn) > 1e-10 else 1e-6
            if abs(rn - sn) <= tol:
                final_bonus = 0.15

        score = min((ratio * 0.85 + final_bonus) * 100, 100.0)
        return {
            "score": round(score, 2),
            "detail": {
                "method": "numeric_comparison",
                "matched": matched,
                "total_ref": len(ref_nums),
                "ratio": round(ratio, 3),
                "final_answer_bonus": final_bonus > 0,
            }
        }


class _CodeScorer:
    """AST node overlap + identifier similarity + syntax validity + length ratio."""

    def _code_block(self, text: str) -> str:
        fence = re.search(r"```(?:python)?\n?(.*?)```", text, re.DOTALL | re.I)
        return fence.group(1).strip() if fence else text.strip()

    def _ast_nodes(self, code: str) -> set:
        try:
            return {type(n).__name__ for n in ast.walk(ast.parse(code))}
        except SyntaxError:
            return set()

    def _identifiers(self, code: str) -> set:
        try:
            return {n.id for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Name)}
        except SyntaxError:
            return set()

    def score(self, student: str, reference: str) -> Dict[str, Any]:
        stud_code = self._code_block(student)
        ref_code  = self._code_block(reference)
        ref_nodes  = self._ast_nodes(ref_code)
        stud_nodes = self._ast_nodes(stud_code)
        ref_ids    = self._identifiers(ref_code)
        stud_ids   = self._identifiers(stud_code)

        if not ref_nodes:
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            sw = set(stopwords.words("english"))
            rw  = {w.lower() for w in word_tokenize(reference) if w.isalpha() and w.lower() not in sw}
            sw2 = {w.lower() for w in word_tokenize(student) if w.isalpha() and w.lower() not in sw}
            ov = len(rw & sw2) / len(rw) if rw else 0.0
            return {"score": round(ov * 100, 2), "detail": {"method": "text_fallback"}}

        node_sim = len(ref_nodes & stud_nodes) / len(ref_nodes)
        id_sim   = len(ref_ids & stud_ids) / len(ref_ids) if ref_ids else 0.5
        syntax   = 0.10 if stud_nodes else 0.0
        ref_lines  = len([l for l in ref_code.splitlines() if l.strip()])
        stud_lines = len([l for l in stud_code.splitlines() if l.strip()])
        length_r   = min(stud_lines / ref_lines, 1.0) if ref_lines else 0.5

        score = (node_sim * 0.50 + id_sim * 0.20 + syntax + length_r * 0.20) * 100
        return {
            "score": round(min(score, 100.0), 2),
            "detail": {
                "ast_node_similarity_pct": round(node_sim * 100, 2),
                "identifier_similarity_pct": round(id_sim * 100, 2),
                "syntax_valid": bool(stud_nodes),
                "ref_lines": ref_lines,
                "student_lines": stud_lines,
            }
        }


class _ListScorer:
    """Set coverage of expected list items via lemma matching."""

    def _items(self, text: str) -> List[str]:
        lines = text.strip().splitlines()
        items = [re.sub(r"^\s*[\-\*\d]+[.)]\s*", "", l).strip() for l in lines]
        items = [i for i in items if i]
        if not items:
            from nltk.tokenize import sent_tokenize
            items = sent_tokenize(text)
        return [i.lower() for i in items]

    def score(self, student: str, reference: str) -> Dict[str, Any]:
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords

        lem = WordNetLemmatizer()
        sw  = set(stopwords.words("english"))

        def kl(text):
            return {lem.lemmatize(w) for w in word_tokenize(text.lower())
                    if w.isalpha() and w not in sw and len(w) > 2}

        ref_items  = self._items(reference)
        stud_items = self._items(student)

        if not ref_items:
            return {"score": 50.0, "detail": {"method": "no_ref_items"}}

        matched, matched_list = 0, []
        for ri in ref_items:
            ri_kl = kl(ri)
            best  = max((len(ri_kl & kl(si)) / len(ri_kl) for si in stud_items if ri_kl), default=0.0)
            if best >= 0.5:
                matched += 1
                matched_list.append(ri)

        coverage    = matched / len(ref_items)
        extra_bonus = min((len(stud_items) - len(ref_items)) * 0.02, 0.10) if len(stud_items) > len(ref_items) else 0.0
        score       = min((coverage + extra_bonus) * 100, 100.0)

        return {
            "score": round(score, 2),
            "detail": {
                "ref_item_count": len(ref_items),
                "student_item_count": len(stud_items),
                "matched_items": matched,
                "coverage_pct": round(coverage * 100, 2),
                "matched_examples": matched_list[:5],
            }
        }


# ── Groq-assisted fallback ─────────────────────────────────────────────────────

def _groq_score(q_type, question, student, reference, groq_client, groq_model):
    import json as _json
    prompt = (
        f"You are an expert academic grader. Score the student answer on a 0-100 scale.\n\n"
        f"Question type: {q_type}\nQuestion: {question}\n\n"
        f"Reference answer:\n{reference[:1500]}\n\nStudent answer:\n{student[:1500]}\n\n"
        'Return ONLY valid JSON: {"score": <int 0-100>, "reason": "<one sentence>"}'
    )
    try:
        resp = groq_client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=120,
        )
        raw  = re.sub(r"^```(?:json)?|```$", "", resp.choices[0].message.content.strip(), flags=re.MULTILINE).strip()
        data = _json.loads(raw)
        return {"score": float(data["score"]), "detail": {"method": "groq", "reason": data.get("reason", "")}}
    except Exception as e:
        logger.warning("Groq scoring fallback failed: %s", e)
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

@dataclass
class RoutedScore:
    question_type: str
    score: float           # 0-100
    detail: Dict[str, Any]
    used_groq_fallback: bool = False


class RoutedScorer:
    """
    Single entry point: classify -> specialised scorer -> optional Groq fallback.

    Usage:
        rs = RoutedScorer(groq_client=groq.client, groq_model="llama-3.3-70b-versatile")
        result = rs.score(question, student_answer, reference_answer)
        # result.question_type, result.score (0-100), result.detail
    """

    GROQ_FALLBACK_THRESHOLD = 30.0  # call Groq when local score < this

    def __init__(self, groq_client=None, groq_model="llama-3.3-70b-versatile", use_groq_fallback=True):
        self.classifier       = QuestionTypeClassifier()

        # v3: use enhanced sympy/set-coverage scorers when available
        try:
            from modules.numerical_sympy_scorer import NumericalSympyScorer
            _num_scorer = NumericalSympyScorer()
        except Exception:
            _num_scorer = _NumericalScorer()

        try:
            from modules.list_set_coverage import ListSetCoverageScorer
            _list_scorer = ListSetCoverageScorer()
        except Exception:
            _list_scorer = _ListScorer()

        self._scorers         = {
            FACTUAL:    _FactualScorer(),
            CONCEPTUAL: _ConceptualScorer(),
            ANALYTICAL: _AnalyticalScorer(),
            NUMERICAL:  _num_scorer,
            CODE:       _CodeScorer(),
            LIST_TYPE:  _list_scorer,
        }
        self.groq_client       = groq_client
        self.groq_model        = groq_model
        self.use_groq_fallback = use_groq_fallback

    def score(self, question: str, student_answer: str, reference_answer: str) -> RoutedScore:
        q_type = self.classifier.classify(question)
        try:
            result = self._scorers[q_type].score(student_answer, reference_answer)
        except Exception as e:
            logger.error("Scorer %s failed: %s", q_type, e)
            result = {"score": 0.0, "detail": {"error": str(e)}}

        used_groq = False
        if self.use_groq_fallback and self.groq_client and result["score"] < self.GROQ_FALLBACK_THRESHOLD:
            fb = _groq_score(q_type, question, student_answer, reference_answer, self.groq_client, self.groq_model)
            if fb:
                blended = fb["score"] * 0.60 + result["score"] * 0.40
                result  = {"score": round(blended, 2), "detail": {**result["detail"], "groq": fb["detail"]}}
                used_groq = True

        return RoutedScore(question_type=q_type, score=round(result["score"], 2),
                           detail=result["detail"], used_groq_fallback=used_groq)

    def score_all(self, qa_triples: List[tuple]) -> List[RoutedScore]:
        """Score a list of (question, student_answer, reference_answer) tuples."""
        return [self.score(q, sa, ra) for q, sa, ra in qa_triples]