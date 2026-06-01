"""
NumericalSympyScorer
--------------------
Robust mathematical equation / expression matching using SymPy.

Features
--------
* Exact symbolic equivalence check  (e.g. x**2 + 2*x == x*(x+2))
* Numerical substitution fallback   (evaluates at random points)
* LaTeX → SymPy conversion          (handles \\frac, \\sqrt, ^, etc.)
* Unit-stripped comparison          (100 km/h == 100)
* Multi-answer extraction from text (finds the LAST equals-expression)
* Output match bonus                (final answer carries extra weight)
"""

import re
import logging
import random
from typing import Optional, Tuple, List, Dict, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LaTeX / plain-text → SymPy-compatible string helpers
# ---------------------------------------------------------------------------

def _latex_to_sympy(text: str) -> str:
    """Best-effort conversion of LaTeX notation to sympy-parseable string."""
    t = text
    # Fractions
    t = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'((\1)/(\2))', t)
    # Sqrt
    t = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', t)
    t = re.sub(r'\\sqrt\s+(\w+)', r'sqrt(\1)', t)
    # Powers written with ^
    t = t.replace('^', '**')
    # Remove LaTeX formatting commands that don't affect value
    t = re.sub(r'\\(?:text|mathrm|mathbf|mathit|left|right)\{([^}]*)\}', r'\1', t)
    t = re.sub(r'\\(?:cdot|times)', '*', t)
    t = re.sub(r'\\div', '/', t)
    t = re.sub(r'\\pm', '+', t)          # simplification — keep +
    t = re.sub(r'\\[a-zA-Z]+', '', t)    # drop remaining commands
    t = re.sub(r'[{}]', '', t)
    return t.strip()


def _extract_rhs_expressions(text: str) -> List[str]:
    """
    Find all "= <expr>" groups in text and return the right-hand sides.
    Also collects standalone numbers/expressions if no = found.
    """
    # Look for   = <stuff>   not followed by another = on the same token
    rhs_list = re.findall(r'=\s*([-\d\.eE+*/^()\s\w\\{},]+?)(?=\s*(?:[=\n]|$))', text)
    if rhs_list:
        return [r.strip() for r in rhs_list if r.strip()]
    # Fallback: grab all numeric-ish tokens
    return re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', text)


def _try_sympify(expr_str: str):
    """Return a sympy expression or None on failure."""
    try:
        import sympy
        return sympy.sympify(expr_str)
    except Exception:
        return None


def _numerical_sample_check(ref_expr, stud_expr, n_samples: int = 8, tol: float = 1e-4) -> Optional[float]:
    """
    Evaluate both expressions at n_samples random points and return
    the fraction of samples that agree within tolerance.
    Uses sympy free symbols; if none, evaluates as constants.
    """
    try:
        import sympy
        free = ref_expr.free_symbols | stud_expr.free_symbols
        matches = 0
        for _ in range(n_samples):
            subs = {s: random.uniform(1, 5) for s in free}
            try:
                rv = complex(ref_expr.subs(subs))
                sv = complex(stud_expr.subs(subs))
                if abs(rv) > 1e-10:
                    if abs(rv - sv) / abs(rv) < tol:
                        matches += 1
                elif abs(sv) < tol:
                    matches += 1
            except Exception:
                pass
        return matches / n_samples
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class NumericalSympyScorer:
    """
    Drop-in replacement / enhancement for the basic _NumericalScorer.

    Usage
    -----
        scorer = NumericalSympyScorer()
        result = scorer.score(student_text, reference_text)
        # result['score']  0-100
        # result['detail'] dict with method, match quality, etc.
    """

    def score(self, student: str, reference: str) -> Dict[str, Any]:
        """
        Compare student answer to reference numerically/symbolically.
        Returns dict with 'score' (0-100) and 'detail'.
        """
        # Pre-process both texts
        ref_clean   = _latex_to_sympy(reference)
        stud_clean  = _latex_to_sympy(student)

        ref_exprs   = _extract_rhs_expressions(ref_clean)
        stud_exprs  = _extract_rhs_expressions(stud_clean)

        if not ref_exprs:
            return self._word_fallback(student, reference)

        # Try symbolic / numerical match on the LAST (final answer) expression
        ref_last   = ref_exprs[-1]
        stud_last  = stud_exprs[-1] if stud_exprs else ""

        sym_score, method = self._symbolic_match(ref_last, stud_last)
        if sym_score is not None:
            # Also check intermediate steps for partial credit
            step_score = self._intermediate_step_score(ref_exprs, stud_exprs)
            final = sym_score * 0.75 + step_score * 0.25
            return {
                "score": round(min(final * 100, 100.0), 2),
                "detail": {
                    "method": method,
                    "final_answer_match": round(sym_score, 3),
                    "step_score": round(step_score, 3),
                    "ref_final": ref_last,
                    "student_final": stud_last,
                }
            }

        # Fallback: brute-force numeric comparison across all extracted numbers
        return self._numeric_fallback(ref_exprs, stud_exprs, student, reference)

    # ------------------------------------------------------------------
    def _symbolic_match(self, ref_expr_str: str, stud_expr_str: str) -> Tuple[Optional[float], str]:
        """Try symbolic then numerical equivalence. Returns (score 0-1, method_name)."""
        ref_sym  = _try_sympify(ref_expr_str)
        stud_sym = _try_sympify(stud_expr_str)

        if ref_sym is None or stud_sym is None:
            return None, "parse_failed"

        # 1. Exact symbolic equality
        try:
            import sympy
            diff = sympy.simplify(ref_sym - stud_sym)
            if diff == 0:
                return 1.0, "symbolic_exact"
        except Exception:
            pass

        # 2. Numerical sampling
        sample = _numerical_sample_check(ref_sym, stud_sym)
        if sample is not None:
            return sample, "numerical_sample"

        # 3. Direct float comparison (constant expressions)
        try:
            rv = float(ref_sym)
            sv = float(stud_sym)
            tol = abs(rv) * 0.01 if abs(rv) > 1e-10 else 1e-6
            similarity = max(1.0 - abs(rv - sv) / (abs(rv) + 1e-10), 0.0)
            return similarity, "float_compare"
        except Exception:
            pass

        return None, "failed"

    def _intermediate_step_score(self, ref_exprs: List[str], stud_exprs: List[str]) -> float:
        """Fraction of reference intermediate expressions matched in student answer."""
        if len(ref_exprs) <= 1:
            return 1.0  # no intermediate steps to check
        ref_mid   = ref_exprs[:-1]
        stud_set  = set(stud_exprs)
        matched   = 0
        for re_expr in ref_mid:
            # Direct string match or symbolic
            if re_expr in stud_set:
                matched += 1
                continue
            re_sym = _try_sympify(re_expr)
            if re_sym is None:
                continue
            for st_expr in stud_exprs:
                st_sym = _try_sympify(st_expr)
                if st_sym is None:
                    continue
                try:
                    import sympy
                    if sympy.simplify(re_sym - st_sym) == 0:
                        matched += 1
                        break
                except Exception:
                    pass
        return matched / len(ref_mid) if ref_mid else 1.0

    def _numeric_fallback(self, ref_nums: List[str], stud_nums: List[str],
                          student: str, reference: str) -> Dict[str, Any]:
        """Simple numeric value matching (same as original _NumericalScorer)."""
        def parse_nums(lst):
            results = []
            for r in lst:
                try:
                    results.append(float(r))
                except ValueError:
                    pass
            return results

        ref_floats  = parse_nums(ref_nums)
        stud_floats = parse_nums(stud_nums)

        if not ref_floats:
            return self._word_fallback(student, reference)

        matched = 0
        for rn in ref_floats:
            for sn in stud_floats:
                tol = abs(rn) * 0.01 if abs(rn) > 1e-10 else 1e-6
                if abs(rn - sn) <= tol:
                    matched += 1
                    break

        ratio = matched / len(ref_floats)
        # Final answer bonus
        final_bonus = 0.0
        if ref_floats and stud_floats:
            rn, sn = ref_floats[-1], stud_floats[-1]
            tol = abs(rn) * 0.01 if abs(rn) > 1e-10 else 1e-6
            if abs(rn - sn) <= tol:
                final_bonus = 0.15

        score = min((ratio * 0.85 + final_bonus) * 100, 100.0)
        return {
            "score": round(score, 2),
            "detail": {
                "method": "numeric_fallback",
                "matched": matched,
                "total_ref": len(ref_floats),
                "ratio": round(ratio, 3),
                "final_answer_bonus": final_bonus > 0,
            }
        }

    def _word_fallback(self, student: str, reference: str) -> Dict[str, Any]:
        """Last-resort word overlap for non-numeric questions."""
        try:
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            sw = set(stopwords.words("english"))
        except Exception:
            sw = set()
            word_tokenize = str.split

        rw  = {w.lower() for w in word_tokenize(reference) if w.isalpha() and w.lower() not in sw}
        sw2 = {w.lower() for w in word_tokenize(student)   if w.isalpha() and w.lower() not in sw}
        ov  = len(rw & sw2) / len(rw) if rw else 0.0
        return {"score": round(ov * 100, 2), "detail": {"method": "word_fallback"}}