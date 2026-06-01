import re
import logging
from typing import Dict, List, Any, Tuple
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

class GrammarChecker:

    def __init__(self, is_ocr_source: bool = False):
        self.is_ocr_source = is_ocr_source  # If True, reduce OCR-noise penalties

        # Common grammar error patterns
        self.grammar_patterns = {
            'double_space': (r'\s{2,}', 'Multiple consecutive spaces'),
            'missing_punctuation': (r'[a-zA-Z]\s*\n', 'Missing punctuation at line end'),
            'repeated_punctuation': (r'[.!?]{2,}', 'Repeated punctuation marks'),
            'capitalization_after_period': (r'(?<![A-Za-z]{3})\.\s+[a-z]', 'Missing capitalization after period'),
            'incomplete_sentences': (r'\b(?:and|but|or|because|since|while)\s*[.!?]\s*', 'Incomplete sentence structure'),
            'comma_before_and': (r'\w+\s+and\s+\w+(?=[.!?])', 'Possible missing comma in list'),
            'its_vs_its': (r'\bits\s+(?:not|never|just|only)\b', "Check: should this be \"it's\"?"),
            'their_there': (r'\bthere\s+(?:are|is)\s+\w+\s+(?:house|car|book|dog|cat)', 'Possible their/there confusion'),
            'your_youre': (r'\byour\s+(?:going|coming|being)', "Check: should this be \"you're\"?"),
            'then_than': (r'\b(?:better|worse|more|less)\s+then\b', 'Should be "than" not "then"'),
            'affect_effect': (r'\beffect\s+(?:the|my|his|her|their)', 'Should likely be "affect" (verb)'),
            'could_of': (r'\b(?:could|should|would)\s+of\b', 'Should be "could have" or "could\'ve"'),
            'alot': (r'\balot\b', 'Should be "a lot" (two words)'),
            'loose_lose': (r'\bloose\s+(?:the|my|his|her|a)', 'Check: should this be "lose"?'),
            'lead_led': (r'\blead\s+(?:to|the|them)', 'Check tense: "lead" (present) or "led" (past)?'),
        }

        # Subject-verb agreement patterns
        self.sv_agreement_patterns = [
            (r'\b(?:he|she|it)\s+(?:are|were|have)\b', 'Subject-verb agreement: singular subject with plural verb'),
            (r'\b(?:they|we)\s+(?:is|was|has)\b', 'Subject-verb agreement: plural subject with singular verb'),
            (r'\b(?:everyone|someone|anyone|everybody)\s+(?:are|were|have)\b', 'Indefinite pronouns are singular'),
        ]

        # Sentence structure issues
        self.structure_patterns = [
            (r'^\s*(?:And|But|Or|Because|Since|While)\s+', 'Sentence starts with conjunction'),
            (r'[.!?]\s*[a-z]', 'Lowercase after sentence end'),
            (r'\b(\w+)\s+\1\b', 'Repeated word'),
        ]

        # Punctuation patterns
        self.punctuation_patterns = [
            (r'[,;:]\s*[,;:]', 'Double punctuation'),
            (r'\s+[,;:.!?]', 'Space before punctuation'),
            (r'[.!?][a-zA-Z]', 'Missing space after punctuation'),
            (r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+', 'Possible over-capitalization'),
        ]
        
    def detect_grammar_errors(self, text: str) -> List[Dict[str, Any]]:
        """Detect various grammar errors in text"""
        errors = []
        
        # Check basic grammar patterns
        for error_type, (pattern, description) in self.grammar_patterns.items():
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                errors.append({
                    'type': error_type,
                    'severity': 'medium',
                    'description': description,
                    'position': match.start(),
                    'text': match.group(0),
                    'category': 'spelling/grammar'
                })
        
        # Check subject-verb agreement
        for pattern, description in self.sv_agreement_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                errors.append({
                    'type': 'subject_verb_agreement',
                    'severity': 'high',
                    'description': description,
                    'position': match.start(),
                    'text': match.group(0),
                    'category': 'grammar'
                })
        
        # Check sentence structure
        for pattern, description in self.structure_patterns:
            matches = list(re.finditer(pattern, text, re.MULTILINE))
            for match in matches:
                errors.append({
                    'type': 'structure',
                    'severity': 'low',
                    'description': description,
                    'position': match.start(),
                    'text': match.group(0),
                    'category': 'style'
                })
        
        # Check punctuation
        for pattern, description in self.punctuation_patterns:
            matches = list(re.finditer(pattern, text))
            for match in matches:
                errors.append({
                    'type': 'punctuation',
                    'severity': 'medium',
                    'description': description,
                    'position': match.start(),
                    'text': match.group(0),
                    'category': 'punctuation'
                })
        
        return errors
    
    def analyze_sentence_structure(self, text: str) -> Dict[str, Any]:
       
        sentences = sent_tokenize(text)
        
        if not sentences:
            return {
                'avg_length': 0,
                'length_variation': 0,
                'simple_sentences': 0,
                'complex_sentences': 0,
                'compound_sentences': 0,
                'sentence_variety_score': 0
            }
        
        sentence_lengths = []
        simple_count = 0
        complex_count = 0
        compound_count = 0
        
        for sentence in sentences:
            words = word_tokenize(sentence)
            sentence_lengths.append(len(words))
            
            # Classify sentence type
            has_conjunction = bool(re.search(r'\b(?:and|but|or|nor|yet|so)\b', sentence.lower()))
            has_subordination = bool(re.search(r'\b(?:because|although|since|while|if|when|unless|until)\b', sentence.lower()))
            has_comma = ',' in sentence
            
            if has_subordination:
                complex_count += 1
            elif has_conjunction and has_comma:
                compound_count += 1
            else:
                simple_count += 1
        
        avg_length = np.mean(sentence_lengths)
        length_variation = np.std(sentence_lengths)
        
        # Calculate variety score
        total = len(sentences)
        variety_score = (
            min(simple_count / total, 0.4) * 40 +  # Up to 40% simple is good
            min(complex_count / total, 0.4) * 40 +  # Up to 40% complex is good
            min(compound_count / total, 0.3) * 20   # Up to 30% compound is good
        ) * 100
        
        return {
            'avg_length': round(avg_length, 2),
            'length_variation': round(length_variation, 2),
            'simple_sentences': simple_count,
            'complex_sentences': complex_count,
            'compound_sentences': compound_count,
            'sentence_variety_score': round(variety_score, 2),
            'total_sentences': total
        }
    
    def analyze_vocabulary(self, text: str) -> Dict[str, Any]:
        """Advanced vocabulary analysis"""
        words = word_tokenize(text)
        words_alpha = [w.lower() for w in words if w.isalpha()]
        
        if not words_alpha:
            return {
                'diversity': 0,
                'sophistication': 0,
                'unique_words': 0,
                'total_words': 0,
                'repeated_words': []
            }
        
        word_freq = Counter(words_alpha)
        unique_words = len(word_freq)
        total_words = len(words_alpha)
        
        # Lexical diversity (Type-Token Ratio)
        diversity = unique_words / total_words
        
        # Find repeated words (used more than expected)
        avg_frequency = total_words / unique_words
        overused_words = [
            (word, count) for word, count in word_freq.most_common(20)
            if count > avg_frequency * 2 and len(word) > 3
        ]
        
        # Vocabulary sophistication (based on word length as proxy)
        avg_word_length = np.mean([len(w) for w in words_alpha])
        sophistication = min((avg_word_length - 3) / 7 * 100, 100)  # 3-10 letter range
        
        return {
            'diversity': round(diversity, 3),
            'sophistication': round(sophistication, 2),
            'unique_words': unique_words,
            'total_words': total_words,
            'avg_word_length': round(avg_word_length, 2),
            'overused_words': overused_words[:5],
            'vocabulary_richness': round(diversity * 100, 2)
        }
    
    def analyze_readability(self, text: str) -> Dict[str, Any]:
        
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        words_alpha = [w for w in words if w.isalpha()]
        
        if not sentences or not words_alpha:
            return {
                'flesch_reading_ease': 0,
                'readability_level': 'N/A',
                'reading_difficulty': 'Cannot calculate'
            }
        
        # Count syllables (approximation)
        def count_syllables(word):
            word = word.lower()
            vowels = 'aeiou'
            syllable_count = 0
            previous_was_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not previous_was_vowel:
                    syllable_count += 1
                previous_was_vowel = is_vowel
            
            # Adjust for silent e
            if word.endswith('e'):
                syllable_count -= 1
            
            return max(1, syllable_count)
        
        total_syllables = sum(count_syllables(word) for word in words_alpha)
        
        # Flesch Reading Ease
        avg_sentence_length = len(words_alpha) / len(sentences)
        avg_syllables_per_word = total_syllables / len(words_alpha)
        
        flesch_score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        flesch_score = max(0, min(100, flesch_score))
        
        # Determine reading level
        if flesch_score >= 90:
            level = 'Very Easy (5th grade)'
        elif flesch_score >= 80:
            level = 'Easy (6th grade)'
        elif flesch_score >= 70:
            level = 'Fairly Easy (7th grade)'
        elif flesch_score >= 60:
            level = 'Standard (8th-9th grade)'
        elif flesch_score >= 50:
            level = 'Fairly Difficult (10th-12th grade)'
        elif flesch_score >= 30:
            level = 'Difficult (College)'
        else:
            level = 'Very Difficult (Graduate)'
        
        return {
            'flesch_reading_ease': round(flesch_score, 2),
            'readability_level': level,
            'avg_sentence_length': round(avg_sentence_length, 2),
            'avg_syllables_per_word': round(avg_syllables_per_word, 2),
            'total_syllables': total_syllables
        }
    
    def check_consistency(self, text: str) -> Dict[str, Any]:
        """Check for consistency in writing"""
        issues = []
        
        # Check tense consistency
        sentences = sent_tokenize(text)
        tenses = []
        
        for sentence in sentences:
            words = word_tokenize(sentence)
            tagged = pos_tag(words)
            
            # Count verb tenses
            past_verbs = sum(1 for word, tag in tagged if tag in ['VBD', 'VBN'])
            present_verbs = sum(1 for word, tag in tagged if tag in ['VBP', 'VBZ'])
            
            if past_verbs > present_verbs:
                tenses.append('past')
            elif present_verbs > past_verbs:
                tenses.append('present')
            else:
                tenses.append('neutral')
        
        # Calculate tense consistency
        if tenses:
            tense_counter = Counter(tenses)
            dominant_tense = tense_counter.most_common(1)[0][0]
            consistency = tense_counter[dominant_tense] / len(tenses)
            
            if consistency < 0.7:
                issues.append({
                    'type': 'tense_inconsistency',
                    'description': 'Inconsistent verb tense usage',
                    'severity': 'medium'
                })
        else:
            consistency = 1.0
        
        # Check punctuation consistency
        quote_styles = {
            'double': len(re.findall(r'"[^"]*"', text)),
            'single': len(re.findall(r"'[^']*'", text))
        }
        
        if quote_styles['double'] > 0 and quote_styles['single'] > 0:
            if abs(quote_styles['double'] - quote_styles['single']) < 2:
                issues.append({
                    'type': 'quote_inconsistency',
                    'description': 'Inconsistent quotation mark usage',
                    'severity': 'low'
                })
        
        return {
            'tense_consistency': round(consistency * 100, 2),
            'dominant_tense': tenses[0] if tenses else 'unknown',
            'issues': issues
        }
    
    def _is_non_prose(self, text: str) -> bool:
        """
        Detect if the text is predominantly non-prose (matrices, equations, code, numbers).
        Grammar scoring is not meaningful for such content.
        """
        tokens = text.split()
        if not tokens:
            return False
        # Count tokens that are purely numeric, operators, or matrix-like
        non_prose_tokens = sum(
            1 for t in tokens
            if re.match(r'^[\d\.\+\-\*/=<>≤≥→:,\(\)\[\]]+$', t)
            or re.match(r'^[A-Z]\d*$', t)  # P0, P1, A, B, C matrix labels
            or t in ('True', 'False', 'true', 'false', '→', '≤', '≥', '∴', '∵')
        )
        ratio = non_prose_tokens / len(tokens)
        # Also check if alpha words are very sparse
        alpha_tokens = sum(1 for t in tokens if re.match(r'^[a-zA-Z]{3,}$', t))
        alpha_ratio = alpha_tokens / len(tokens)
        return ratio > 0.35 or alpha_ratio < 0.25

    def analyze_grammar_and_style(self, text: str) -> Dict[str, Any]:
        """Comprehensive grammar and style analysis"""

        # If the text is predominantly numerical/matrix/code, grammar scoring
        # is not meaningful — return a neutral score of 50 with a flag.
        if self._is_non_prose(text):
            return {
                'score': 50.0,
                'total_errors': 0,
                'errors': [],
                'errors_by_category': {},
                'errors_by_severity': {'high': 0, 'medium': 0, 'low': 0},
                'sentence_analysis': {'avg_length': 0, 'sentence_variety_score': 0, 'total_sentences': 0},
                'vocabulary_analysis': {'diversity': 0, 'total_words': len(text.split()), 'unique_words': 0, 'vocabulary_richness': 0},
                'readability': {'flesch_reading_ease': 50, 'readability_level': 'N/A (non-prose content)'},
                'consistency': {'tense_consistency': 100, 'dominant_tense': 'N/A', 'issues': []},
                'non_prose': True,
            }
        
        # Detect errors
        errors = self.detect_grammar_errors(text)
        
        # Analyze components
        sentence_analysis = self.analyze_sentence_structure(text)
        vocab_analysis = self.analyze_vocabulary(text)
        readability = self.analyze_readability(text)
        consistency = self.check_consistency(text)
        
        # Calculate scores by severity
        error_penalties = {
            'high': 5,
            'medium': 3,
            'low': 1
        }
        
        total_penalty = sum(error_penalties.get(e['severity'], 2) for e in errors)
        
        # Base grammar score
        grammar_score = 100.0
        
        # Apply error penalties
        word_count = len(word_tokenize(text))
        if word_count > 0:
            error_density_penalty = min((total_penalty / word_count) * 100, 50)
            grammar_score -= (error_density_penalty * 0.5)
        
        # Adjust based on sentence variety
        if sentence_analysis['sentence_variety_score'] > 60:
            grammar_score += 5
        
        # Adjust based on vocabulary
        if vocab_analysis['diversity'] > 0.6:
            grammar_score += 5
        elif vocab_analysis['diversity'] < 0.3:
            grammar_score -= 5
        
        # Adjust based on consistency
        if consistency['tense_consistency'] > 85:
            grammar_score += 3
        elif consistency['tense_consistency'] < 60:
            grammar_score -= 5
        
        # Ensure score is in valid range
        grammar_score = max(0, min(100, grammar_score))
        
        # Categorize errors
        errors_by_category = defaultdict(list)
        for error in errors:
            errors_by_category[error['category']].append(error)
        
        return {
            'score': round(grammar_score, 2),
            'total_errors': len(errors),
            'errors': errors[:10],  # Top 10 errors
            'errors_by_category': {k: len(v) for k, v in errors_by_category.items()},
            'errors_by_severity': {
                'high': sum(1 for e in errors if e['severity'] == 'high'),
                'medium': sum(1 for e in errors if e['severity'] == 'medium'),
                'low': sum(1 for e in errors if e['severity'] == 'low')
            },
            'sentence_analysis': sentence_analysis,
            'vocabulary_analysis': vocab_analysis,
            'readability': readability,
            'consistency': consistency
        }
    
    def get_writing_quality_feedback(self, analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate comprehensive writing quality feedback"""
        feedback = {
            'strengths': [],
            'weaknesses': [],
            'suggestions': []
        }
        
        # Grammar errors feedback
        total_errors = analysis['total_errors']
        if total_errors == 0:
            feedback['strengths'].append("✓ Excellent grammar with no errors detected")
        elif total_errors <= 3:
            feedback['strengths'].append("✓ Good grammar with only minor issues")
        else:
            feedback['weaknesses'].append(f"✗ Grammar needs improvement ({total_errors} issues found)")
            
            # Specific error feedback
            high_severity = analysis['errors_by_severity']['high']
            if high_severity > 0:
                feedback['suggestions'].append(f"→ Fix {high_severity} critical grammar error(s) first")
        
        # Sentence structure feedback
        sent_analysis = analysis['sentence_analysis']
        if sent_analysis['sentence_variety_score'] > 70:
            feedback['strengths'].append("✓ Good sentence variety and structure")
        elif sent_analysis['sentence_variety_score'] < 40:
            feedback['weaknesses'].append("✗ Limited sentence variety")
            feedback['suggestions'].append("→ Mix simple, compound, and complex sentences")
        
        avg_length = sent_analysis['avg_length']
        if 12 <= avg_length <= 20:
            feedback['strengths'].append(f"✓ Well-balanced sentence length (avg: {avg_length} words)")
        elif avg_length < 8:
            feedback['suggestions'].append("→ Consider adding more detail to sentences")
        elif avg_length > 25:
            feedback['suggestions'].append("→ Break down longer sentences for clarity")
        
        # Vocabulary feedback
        vocab = analysis['vocabulary_analysis']
        if vocab['diversity'] > 0.65:
            feedback['strengths'].append(f"✓ Rich vocabulary (diversity: {vocab['vocabulary_richness']}%)")
        elif vocab['diversity'] < 0.35:
            feedback['weaknesses'].append("✗ Limited vocabulary variety")
            feedback['suggestions'].append("→ Use more varied words and avoid repetition")
        
        if vocab['overused_words']:
            overused = ', '.join([f"'{word}'" for word, _ in vocab['overused_words'][:3]])
            feedback['suggestions'].append(f"→ Words used frequently: {overused} - consider alternatives")
        
        # Readability feedback
        readability = analysis['readability']
        flesch = readability['flesch_reading_ease']
        if 60 <= flesch <= 80:
            feedback['strengths'].append(f"✓ Good readability level ({readability['readability_level']})")
        elif flesch < 30:
            feedback['suggestions'].append("→ Text is quite complex - consider simplifying for broader audience")
        elif flesch > 90:
            feedback['suggestions'].append("→ Text is very simple - consider adding more depth")
        
        # Consistency feedback
        consistency = analysis['consistency']
        if consistency['tense_consistency'] > 85:
            feedback['strengths'].append("✓ Consistent verb tense usage")
        elif consistency['tense_consistency'] < 60:
            feedback['weaknesses'].append("✗ Inconsistent verb tense")
            feedback['suggestions'].append(f"→ Maintain consistent tense (primarily {consistency['dominant_tense']})")
        
        return feedback