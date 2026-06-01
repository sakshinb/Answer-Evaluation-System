import logging
import re
from typing import Dict, List, Any, Set, Tuple, Optional
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from collections import defaultdict
SYNONYMS = {
    "authentication": [
        "login verification",
        "identity validation"
    ],

    "anomaly detection": [
        "threat detection",
        "suspicious activity detection"
    ],

    "alert": [
        "notification",
        "warning"
    ]
}

logger = logging.getLogger(__name__)

class FactChecker:
   
    
    def __init__(self):
        
        # Relationship patterns for fact extraction
        self.relationship_patterns = [
            # Capital relationships
            (r'capital\s+of\s+(\w+(?:\s+\w+)?)\s+is\s+(\w+(?:\s+\w+)?)', 'capital_of'),
            (r'(\w+(?:\s+\w+)?)\s+is\s+the\s+capital\s+of\s+(\w+(?:\s+\w+)?)', 'capital_of_reverse'),
            
            # Location relationships  
            (r'(\w+(?:\s+\w+)?)\s+is\s+(?:in|located in)\s+(\w+(?:\s+\w+)?)', 'located_in'),
            (r'(\w+(?:\s+\w+)?)\s+is\s+a\s+(?:city|country|state|province)\s+in\s+(\w+(?:\s+\w+)?)', 'located_in'),
            
            # Inventor/Creator relationships
            (r'(\w+(?:\s+\w+)?)\s+(?:invented|created|discovered)\s+(?:the\s+)?(\w+(?:\s+\w+)?)', 'invented_by'),
            (r'(?:the\s+)?(\w+(?:\s+\w+)?)\s+was\s+(?:invented|created|discovered)\s+by\s+(\w+(?:\s+\w+)?)', 'invented_by'),
            
            # Author/Artist relationships
            (r'(\w+(?:\s+\w+)?)\s+(?:wrote|authored|composed)\s+(?:the\s+)?["\']?(\w+(?:\s+\w+)?)["\']?', 'authored_by'),
            (r'["\']?(\w+(?:\s+\w+)?)["\']?\s+was\s+written\s+by\s+(\w+(?:\s+\w+)?)', 'authored_by'),
            
            # Born in relationships
            (r'(\w+(?:\s+\w+)?)\s+was\s+born\s+in\s+(\w+(?:\s+\w+)?)', 'born_in'),
            (r'(\w+(?:\s+\w+)?)\s+born\s+(?:in\s+)?(\d{4})', 'born_year'),
            
            # Is-a relationships
            (r'(\w+(?:\s+\w+)?)\s+is\s+a\s+(\w+(?:\s+\w+)?)', 'is_a'),
            
            # Measurement relationships
            (r'(\w+(?:\s+\w+)?)\s+is\s+(\d+\.?\d*)\s*(\w+)', 'measurement'),
            
            # Date relationships
            (r'(\w+(?:\s+\w+)?)\s+(?:happened|occurred)\s+in\s+(\d{4})', 'year_of'),
            (r'in\s+(\d{4}),?\s+(\w+(?:\s+\w+)?)', 'year_of_reverse'),
        ]
        
    def extract_factual_relationships(self, text: str) -> List[Dict[str, Any]]:
       
        facts = []
        text_lower = text.lower()
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            for pattern, relation_type in self.relationship_patterns:
                matches = re.finditer(pattern, sentence_lower, re.IGNORECASE)
                
                for match in matches:
                    if relation_type == 'capital_of':
                        # "capital of X is Y" -> Y is capital of X
                        country = match.group(1).strip()
                        capital = match.group(2).strip()
                        facts.append({
                            'subject': capital,
                            'relation': 'capital_of',
                            'object': country,
                            'original_sentence': sentence.strip(),
                            'confidence': 'high'
                        })
                    elif relation_type == 'capital_of_reverse':
                        # "X is the capital of Y" -> X is capital of Y
                        capital = match.group(1).strip()
                        country = match.group(2).strip()
                        facts.append({
                            'subject': capital,
                            'relation': 'capital_of',
                            'object': country,
                            'original_sentence': sentence.strip(),
                            'confidence': 'high'
                        })
                    elif relation_type in ['located_in', 'invented_by', 'authored_by', 'born_in', 'is_a']:
                        subject = match.group(1).strip()
                        obj = match.group(2).strip()
                        facts.append({
                            'subject': subject,
                            'relation': relation_type,
                            'object': obj,
                            'original_sentence': sentence.strip(),
                            'confidence': 'high'
                        })
                    elif relation_type == 'measurement':
                        subject = match.group(1).strip()
                        value = match.group(2).strip()
                        unit = match.group(3).strip()
                        facts.append({
                            'subject': subject,
                            'relation': 'measurement',
                            'object': f"{value} {unit}",
                            'original_sentence': sentence.strip(),
                            'confidence': 'high'
                        })
                    elif relation_type in ['year_of', 'year_of_reverse', 'born_year']:
                        if relation_type == 'year_of_reverse':
                            year = match.group(1).strip()
                            event = match.group(2).strip()
                        else:
                            event = match.group(1).strip()
                            year = match.group(2).strip()
                        facts.append({
                            'subject': event,
                            'relation': 'year',
                            'object': year,
                            'original_sentence': sentence.strip(),
                            'confidence': 'high'
                        })
        
        return facts
    
    def compare_factual_relationships(self, student_text: str, reference_text: str) -> Dict[str, Any]:
        
        student_facts = self.extract_factual_relationships(student_text)
        reference_facts = self.extract_factual_relationships(reference_text)
        
        # Normalize facts for comparison
        def normalize_fact(fact):
            return (
                fact['subject'].lower().strip(),
                fact['relation'],
                fact['object'].lower().strip()
            )
        
        student_fact_set = {normalize_fact(f) for f in student_facts}
        reference_fact_set = {normalize_fact(f) for f in reference_facts}
        
        # Find matches and mismatches
        correct_facts = student_fact_set & reference_fact_set
        missing_facts = reference_fact_set - student_fact_set
        
        # Find contradictions - same subject and relation but different object
        contradictions = []
        for stud_fact in student_facts:
            stud_norm = normalize_fact(stud_fact)
            
            for ref_fact in reference_facts:
                ref_norm = normalize_fact(ref_fact)
                
                # Same subject and relation but different object = contradiction
                if (stud_norm[0] == ref_norm[0] and  # same subject
                    stud_norm[1] == ref_norm[1] and  # same relation
                    stud_norm[2] != ref_norm[2]):    # different object
                    
                    contradictions.append({
                        'subject': stud_norm[0],
                        'relation': stud_norm[1],
                        'student_answer': stud_norm[2],
                        'correct_answer': ref_norm[2],
                        'student_sentence': stud_fact['original_sentence'],
                        'reference_sentence': ref_fact['original_sentence']
                    })
        
        # Extra facts that weren't in reference (could be wrong or additional info)
        potentially_wrong = student_fact_set - reference_fact_set - {
            (c['subject'], c['relation'], c['student_answer']) 
            for c in contradictions
        }
        
        return {
            'correct_facts': list(correct_facts),
            'missing_facts': list(missing_facts),
            'contradictions': contradictions,
            'potentially_wrong': list(potentially_wrong),
            'student_facts': student_facts,
            'reference_facts': reference_facts
        }
        
    def extract_claims(self, text: str) -> List[Dict[str, Any]]:
        
        sentences = sent_tokenize(text)
        claims = []
        
        # Patterns that typically indicate factual statements
        factual_patterns = [
            (r'\b(is|are|was|were|has|have|had)\b', 'state_of_being'),
            (r'\b(called|named|known as|referred to as)\b', 'definition'),
            (r'\b(\d+\.?\d*)\s*(percent|%|degrees?|years?|times?|meters?|km|miles?)\b', 'quantitative'),
            (r'\b(invented|discovered|created|founded|established)\b', 'historical'),
            (r'\b(always|never|all|none|every|must|cannot)\b', 'absolute'),
            (r'\b(increases|decreases|causes|results in|leads to|affects)\b', 'causal'),
        ]
        
        for i, sentence in enumerate(sentences):
            claim_info = {
                'sentence': sentence,
                'index': i,
                'type': 'general',
                'confidence': 'medium',
                'entities': self._extract_entities(sentence),
                'numbers': self._extract_numbers(sentence)
            }
            
            # Determine claim type based on patterns
            for pattern, claim_type in factual_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    claim_info['type'] = claim_type
                    break
            
            # Assess confidence based on linguistic markers
            claim_info['confidence'] = self._assess_claim_confidence(sentence)
            
            claims.append(claim_info)
        
        return claims
    
    def _normalize_entity(self, entity: str) -> str:
        """Lowercase, strip, collapse whitespace for fuzzy entity comparison."""
        return re.sub(r'\s+', ' ', entity.lower().strip())

    def _entities_match(self, e1: str, e2: str) -> bool:
        """
        Fuzzy entity matching:
        1. Exact normalized match
        2. One is a substring of the other (min 4 chars)
        3. Stem match (first 5 chars of each word)
        """
        n1, n2 = self._normalize_entity(e1), self._normalize_entity(e2)
        if n1 == n2:
            return True
        # Substring containment
        if len(n1) >= 4 and len(n2) >= 4:
            if n1 in n2 or n2 in n1:
                return True
        # Word-level stem match
        words1 = [w[:5] for w in n1.split() if len(w) >= 4]
        words2 = [w[:5] for w in n2.split() if len(w) >= 4]
        if words1 and words2:
            overlap = sum(1 for w in words1 if w in words2)
            if overlap >= max(1, min(len(words1), len(words2)) - 1):
                return True
        return False

    def _fuzzy_entity_match_ratio(self, student_entities: set, reference_entities: set) -> float:
        """
        Compute match ratio using fuzzy matching instead of exact set intersection.
        Each reference entity is considered matched if any student entity fuzzy-matches it.
        """
        if not reference_entities:
            return 1.0
        matched = sum(
            1 for ref_e in reference_entities
            if any(self._entities_match(ref_e, stu_e) for stu_e in student_entities)
        )
        return matched / len(reference_entities)

    def _extract_entities(self, text: str) -> List[str]:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)

        entities = []
        current_entity = []

        for word, tag in tagged:
            # Capture proper nouns AND regular nouns (technical terms like "scheduler", "process")
            if tag.startswith('NNP') or tag.startswith('NN'):
                current_entity.append(word)
            else:
                if current_entity:
                    entity = ' '.join(current_entity)
                    # Only keep entities with meaningful length
                    if len(entity) >= 3:
                        entities.append(entity)
                    current_entity = []

        if current_entity:
            entity = ' '.join(current_entity)
            if len(entity) >= 3:
                entities.append(entity)

        return entities
    
    def _extract_numbers(self, text: str) -> List[Dict[str, str]]:
       
        number_patterns = [
            r'(\d+\.?\d*)\s*(percent|%)',
            r'(\d+\.?\d*)\s*(degrees?|°[CF]?)',
            r'(\d+\.?\d*)\s*(years?|months?|days?|hours?|minutes?|seconds?)',
            r'(\d+\.?\d*)\s*(meters?|km|kilometers?|miles?|feet|inches?)',
            r'(\d+\.?\d*)\s*(kg|kilograms?|pounds?|lbs?|tons?)',
            r'(\d{4})\s*(?:AD|BC|CE|BCE)?',  # Years
        ]
        
        numbers = []
        for pattern in number_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                numbers.append({
                    'value': match.group(1),
                    'unit': match.group(2) if len(match.groups()) > 1 else '',
                    'context': match.group(0)
                })
        
        return numbers
    
    def _assess_claim_confidence(self, sentence: str) -> str:
        
        sentence_lower = sentence.lower()
        
        # High confidence markers
        high_confidence = ['proven', 'demonstrated', 'confirmed', 'established', 
                          'verified', 'documented', 'definitely', 'certainly']
        
        # Low confidence markers
        low_confidence = ['possibly', 'perhaps', 'might', 'may', 'could', 
                         'probably', 'likely', 'seems', 'appears', 'suggests',
                         'approximately', 'around', 'roughly', 'estimated']
        
        if any(marker in sentence_lower for marker in high_confidence):
            return 'high'
        elif any(marker in sentence_lower for marker in low_confidence):
            return 'low'
        else:
            return 'medium'
    
    def compare_claims(self, student_text: str, reference_text: str) -> Dict[str, Any]:
       
        student_claims = self.extract_claims(student_text)
        reference_claims = self.extract_claims(reference_text)
        
        # Extract all entities and numbers from both texts
        student_entities = set()
        student_numbers = set()
        for claim in student_claims:
            student_entities.update(claim['entities'])
            student_numbers.update(n['context'] for n in claim['numbers'])
        
        reference_entities = set()
        reference_numbers = set()
        for claim in reference_claims:
            reference_entities.update(claim['entities'])
            reference_numbers.update(n['context'] for n in claim['numbers'])
        
        # Find matches and mismatches using fuzzy matching
        fuzzy_match_ratio = self._fuzzy_entity_match_ratio(student_entities, reference_entities)
        matching_entities = student_entities & reference_entities  # kept for display
        missing_entities = reference_entities - student_entities
        extra_entities = student_entities - reference_entities
        
        matching_numbers = student_numbers & reference_numbers
        missing_numbers = reference_numbers - student_numbers
        contradicting_numbers = self._find_contradicting_numbers(
            student_numbers, reference_numbers
        )
        
        return {
            'entities': {
                'matching': list(matching_entities),
                'missing': list(missing_entities),
                'extra': list(extra_entities),
                'match_ratio': fuzzy_match_ratio
            },
            'numbers': {
                'matching': list(matching_numbers),
                'missing': list(missing_numbers),
                'contradicting': contradicting_numbers,
                'match_ratio': len(matching_numbers) / len(reference_numbers) if reference_numbers else 1.0
            },
            'claim_types': self._analyze_claim_types(student_claims, reference_claims)
        }
    
    def _find_contradicting_numbers(self, student_numbers: Set[str], 
                                   reference_numbers: Set[str]) -> List[Dict[str, str]]:
        
        contradictions = []
        
        # Extract just the numeric values and units
        def parse_number(num_str):
            match = re.search(r'(\d+\.?\d*)\s*(.+)?', num_str)
            if match:
                return match.group(1), match.group(2) or ''
            return None, None
        
        student_parsed = [parse_number(n) for n in student_numbers]
        reference_parsed = [parse_number(n) for n in reference_numbers]
        
        for ref_val, ref_unit in reference_parsed:
            if ref_val is None:
                continue
            for stud_val, stud_unit in student_parsed:
                if stud_val is None:
                    continue
                # Same unit but different values
                if ref_unit.lower().strip() == stud_unit.lower().strip():
                    if ref_val != stud_val:
                        contradictions.append({
                            'reference': f"{ref_val} {ref_unit}",
                            'student': f"{stud_val} {stud_unit}",
                            'type': 'value_mismatch'
                        })
        
        return contradictions
    
    def _analyze_claim_types(self, student_claims: List[Dict], 
                            reference_claims: List[Dict]) -> Dict[str, Any]:
       
        student_types = defaultdict(int)
        reference_types = defaultdict(int)
        
        for claim in student_claims:
            student_types[claim['type']] += 1
        
        for claim in reference_claims:
            reference_types[claim['type']] += 1
        
        return {
            'student_distribution': dict(student_types),
            'reference_distribution': dict(reference_types),
            'coverage': {
                claim_type: student_types.get(claim_type, 0) / count 
                for claim_type, count in reference_types.items()
            } if reference_types else {}
        }
    
    def verify_factual_accuracy(self, student_text: str, reference_text: str) -> Dict[str, Any]:
        """Comprehensive factual accuracy verification"""
        
        # Method 1: Relationship-based fact checking (most important)
        relationship_analysis = self.compare_factual_relationships(student_text, reference_text)
        
        # Method 2: General claim comparison
        claim_comparison = self.compare_claims(student_text, reference_text)
        
        # Calculate relationship accuracy
        total_ref_facts = len(relationship_analysis['reference_facts'])
        correct_facts = len(relationship_analysis['correct_facts'])
        contradictions = len(relationship_analysis['contradictions'])
        
        if total_ref_facts > 0:
            relationship_score = (correct_facts / total_ref_facts) * 100
            # Reduced from 25 to 10 — OCR noise frequently creates false contradictions
            # (garbled words get regex-matched to wrong fact relationships)
            contradiction_penalty = contradictions * 5  # OCR noise causes false contradictions
            relationship_score = max(relationship_score - contradiction_penalty, 0)
        else:
            # No specific facts to check, use general comparison
            relationship_score = None
        
        # Calculate entity accuracy — apply a floor since fuzzy matching against
        # LLM-generated model answers still misses paraphrased terms.
        entity_score = max(claim_comparison['entities']['match_ratio'] * 100, 55.0)
        
        # Calculate numerical accuracy
        num_match_ratio = claim_comparison['numbers']['match_ratio']
        num_contradictions = len(claim_comparison['numbers']['contradicting'])
        
        # Penalty for numerical contradictions
        num_contradiction_penalty = min(num_contradictions * 15, 50)
        numerical_score = max((num_match_ratio * 100) - num_contradiction_penalty, 0)
        
        # Overall factual score
        if relationship_score is not None:
            # If we have specific facts, prioritize them heavily
            factual_score = (
                relationship_score * 0.4 +  # Relationship facts most important
                entity_score * 0.35 +       # Key terms
                numerical_score * 0.25      # Numbers
            )
        else:
            # Fall back to general comparison.
            factual_score = (entity_score * 0.6 + numerical_score * 0.4)
        
        # Detect absolute claims (which need higher accuracy)
        student_claims = self.extract_claims(student_text)
        absolute_claims = [c for c in student_claims if c['type'] == 'absolute']
        
        if absolute_claims and (claim_comparison['entities']['missing'] or contradictions > 0):
            # Softer penalty (0.92 instead of 0.85) to avoid over-penalizing OCR scans
            factual_score *= 0.92
        
        return {
            'score': min(factual_score, 100.0),
            'relationship_accuracy': round(relationship_score, 2) if relationship_score is not None else None,
            'entity_accuracy': round(entity_score, 2),
            'numerical_accuracy': round(numerical_score, 2),
            'relationship_analysis': relationship_analysis,
            'claim_comparison': claim_comparison,
            'total_student_claims': len(student_claims),
            'total_reference_claims': len(self.extract_claims(reference_text)),
            'absolute_claims': len(absolute_claims),
            'contradictions': relationship_analysis['contradictions'],
            'numerical_contradictions': claim_comparison['numbers']['contradicting']
        }
    
    def get_factual_feedback(self, analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate detailed factual accuracy feedback"""
        feedback = {
            'strengths': [],
            'weaknesses': [],
            'suggestions': []
        }
        
        # Relationship-based fact checking feedback (MOST IMPORTANT)
        if 'relationship_analysis' in analysis and analysis['relationship_analysis']:
            rel_analysis = analysis['relationship_analysis']
            
            # Contradictions - CRITICAL errors
            if rel_analysis['contradictions']:
                feedback['weaknesses'].append(
                    f"❌ Found {len(rel_analysis['contradictions'])} factual error(s):"
                )
                for contradiction in rel_analysis['contradictions'][:3]:
                    feedback['weaknesses'].append(
                        f"   • Wrong: '{contradiction['subject']} {contradiction['relation'].replace('_', ' ')} "
                        f"{contradiction['student_answer']}'"
                    )
                    feedback['suggestions'].append(
                        f"   ✓ Correct: '{contradiction['subject']} {contradiction['relation'].replace('_', ' ')} "
                        f"{contradiction['correct_answer']}'"
                    )
            
            # Correct facts
            if rel_analysis['correct_facts']:
                feedback['strengths'].append(
                    f"✓ Correctly stated {len(rel_analysis['correct_facts'])} factual relationship(s)"
                )
            
            # Missing facts
            if rel_analysis['missing_facts']:
                feedback['weaknesses'].append(
                    f"Missing {len(rel_analysis['missing_facts'])} important fact(s) from reference"
                )
        
        # Entity feedback
        matching = analysis['claim_comparison']['entities']['matching']
        missing = analysis['claim_comparison']['entities']['missing']
        extra = analysis['claim_comparison']['entities']['extra']
        
        if matching:
            feedback['strengths'].append(
                f"Correctly mentioned key terms: {', '.join(matching[:5])}"
                + (f" and {len(matching) - 5} more" if len(matching) > 5 else "")
            )
        
        if missing:
            feedback['weaknesses'].append(
                f"Missing important terms: {', '.join(missing[:5])}"
                + (f" and {len(missing) - 5} more" if len(missing) > 5 else "")
            )
            feedback['suggestions'].append(
                "Include more key concepts and terminology from the topic"
            )
        
        if extra:
            if len(extra) > len(matching):
                feedback['suggestions'].append(
                    "Focus on relevant information; some mentioned terms may be off-topic"
                )
        
        # Numerical feedback
        if 'numerical_contradictions' in analysis:
            num_contradictions = analysis['numerical_contradictions']
            if num_contradictions:
                feedback['weaknesses'].append(
                    f"Numerical inaccuracies found: {len(num_contradictions)} mismatch(es)"
                )
                for contradiction in num_contradictions[:3]:
                    feedback['suggestions'].append(
                        f"Verify: stated '{contradiction['student']}' but should be '{contradiction['reference']}'"
                    )
        
        num_matching = analysis['claim_comparison']['numbers']['matching']
        if num_matching:
            feedback['strengths'].append(
                f"Accurate numerical data ({len(num_matching)} value(s) verified)"
            )
        
        # Absolute claims feedback
        if analysis['absolute_claims'] > 0:
            if missing or (analysis.get('contradictions') and len(analysis['contradictions']) > 0):
                feedback['weaknesses'].append(
                    "Made definitive statements while having factual errors or missing information"
                )
                feedback['suggestions'].append(
                    "Use more cautious language when not completely accurate"
                )
        
        # Overall accuracy feedback
        if analysis['score'] >= 90:
            feedback['strengths'].append("Excellent factual accuracy")
        elif analysis['score'] >= 75:
            feedback['strengths'].append("Good factual accuracy overall")
        elif analysis['score'] < 60:
            feedback['suggestions'].append(
                "Review the reference material to improve factual accuracy"
            )
        
        return feedback