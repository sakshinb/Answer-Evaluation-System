"""
Minimal text preprocessor using only NLTK and basic Python
"""

import re
import os
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import Dict, List


class MinimalPreprocessor:
    """Minimal text preprocessor without heavy dependencies"""

    def __init__(self):
        """Initialize with NLTK resources"""
        self.setup_nltk()

    def setup_nltk(self):
        """Download required NLTK resources"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')

        self.stop_words = set(stopwords.words('english'))

    def clean_ocr_artifacts(self, text: str) -> str:
        """Remove OCR artifacts and noise"""
        # Remove special OCR symbols
        text = re.sub(r'[①②③④⑤⑥⑦⑧⑨⑩℗®©™]', '', text)

        # Remove scattered single letters
        text = re.sub(r'\b[a-zA-Z]\s+[a-zA-Z]\s+[a-zA-Z]\b', '', text)

        # Remove repeated characters (4+ times)
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)

        # Clean up non-ASCII but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\/\\\'\"]', ' ', text)

        return text

    def extract_student_info(self, text: str) -> Dict[str, str]:
        """Extract student information"""
        info = {}

        # Extract name
        name_patterns = [
            r'Name[:\-\s]*([^\n\r]+)',
            r'Student[:\-\s]*([^\n\r]+)'
        ]

        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                name = re.sub(r'[^\w\s]', '', name).strip()
                if len(name) > 2:
                    info['name'] = name
                    break

        # Extract roll number/PRN
        roll_patterns = [
            r'(Roll\s*No\.?|PRN)[:\-\s]*(\d+)',
            r'(\d{10,})',
        ]

        for pattern in roll_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) > 1:
                    info['roll_number'] = match.group(2).strip()
                else:
                    info['roll_number'] = match.group(1).strip()
                break

        # Extract division
        div_match = re.search(r'Div[:\-\s]*([^\n\r]+)', text, re.IGNORECASE)
        if div_match:
            info['division'] = div_match.group(1).strip()

        return info

    def remove_headers(self, text: str) -> str:
        """Remove student info headers and metadata"""
        patterns = [
            r'Name[:\-\s]*[^\n]*',
            r'Roll\s*No\.?[:\-\s]*[^\n]*',
            r'PRN[:\-\s]*[^\n]*',
            r'Div[:\-\s]*[^\n]*',
            r'Date[:\-\s]*[^\n]*',
            r'Page\s*No\.?[:\-\s]*[^\n]*',
            r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}',
        ]

        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        return text

    def normalize_text(self, text: str) -> str:
        """Normalize whitespace and formatting"""
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s+([,.;:!?])', r'\1', text)
        text = re.sub(r'([,.;:!?])\s*', r'\1 ', text)

        return text.strip()

    def correct_common_errors(self, text: str) -> str:
        """Fix common OCR errors — expanded for handwritten scan noise"""
        corrections = {
            # OS / technical terms
            r'\b0S\b': 'OS',
            r'\blinun\b': 'linux',
            r'\bwindorus\b': 'windows',
            r'\b[Ww]er\b': 'user',
            r'\bwers\b': 'users',
            r'\bcystem\b': 'system',
            r'\bsytem\b': 'system',
            r'\bsytems\b': 'systems',
            r'\bkernal\b': 'kernel',
            r'\bkemnel\b': 'kernel',
            r'\bbootlaoder\b': 'bootloader',
            r'\bbootlader\b': 'bootloader',
            r'\bsceduling\b': 'scheduling',
            r'\bschedualing\b': 'scheduling',
            r'\bmulatasking\b': 'multitasking',
            r'\bSimultan\s+truly\b': 'simultaneously',
            r'\bsimultanously\b': 'simultaneously',
            r'\bsimultaniously\b': 'simultaneously',
            r'\bprograming\b': 'programming',
            r'\bprogramm\b': 'program',
            r'\boperarting\b': 'operating',
            r'\boperatng\b': 'operating',
            r'\balgoritton\b': 'algorithm',
            r'\balgorithim\b': 'algorithm',
            r'\balgorithms\b': 'algorithms',
            r'\bcompatability\b': 'compatibility',
            r'\bcompatibilty\b': 'compatibility',
            r'\bgranular\s+level\b': 'granular level',
            r'\bmonalithic\b': 'monolithic',
            r'\bmonoloithic\b': 'monolithic',
            r'\bmonlithic\b': 'monolithic',
            r'\bheirarchical\b': 'hierarchical',
            r'\bhierarchal\b': 'hierarchical',
            r'\bfilesytem\b': 'filesystem',
            r'\bfilesytems\b': 'filesystems',
            r'\bauthentification\b': 'authentication',
            r'\bautherization\b': 'authorization',
            r'\bauthorisation\b': 'authorization',
            r'\bencyrption\b': 'encryption',
            r'\bencription\b': 'encryption',
            r'\bpaginaton\b': 'pagination',
            r'\bsegmentaion\b': 'segmentation',
            r'\bvirtuall\b': 'virtual',
            r'\bprotocall\b': 'protocol',
            r'\bprotocals\b': 'protocols',
            r'\bGRUB\s+Uni\b': 'GRUB Grand Unified',
            r'\bmultiboot\b': 'multiboot',
            r'\brecovery\b': 'recovery',

            # punctuation cleanup
            r'\.{2,}': '.',
            r'\,{2,}': ',',

            # garbled OCR words
            r'\bbouwwhich\b': 'which',
            r'\bbouwhich\b': 'which',
            r'\bconfigur\b': 'configure',
            r'\bconfigue\b': 'configure',
            r'\bconfiguation\b': 'configuration',
            r'\bwer\s+base\b': 'user base',
            r'\bwer\s+friendly\b': 'user-friendly',
            r'\bwer\s+interface\b': 'user interface',
            r'\bcurspecte\b': 'aspect',
            r'\bcurgoste\b': 'component',
            r'\bpessenury\b': 'necessary',
            r'\bpеssenury\b': 'necessary',
            r'\bnecessarу\b': 'necessary',
        }

        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def deep_clean_ocr_text(self, text: str) -> str:
        """
        Additional pass specifically for scanned handwritten documents.
        Fixes structural noise that corrupts grammar and fact checkers.
        """
        # Remove isolated numbers/symbols acting as bullet decorators
        text = re.sub(r'(?<!\w)\d(?!\w|\.\d)', '', text)

        # Collapse lines broken mid-sentence (handwriting line breaks)
        text = re.sub(r'([a-z,])\n([a-z])', r'\1 \2', text)

        # Remove stray single-character lines (OCR noise)
        text = re.sub(r'^\s*[a-zA-Z]\s*$', '', text, flags=re.MULTILINE)

        # Fix words split across lines with a hyphen
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

        # Remove page/date metadata lines that slipped through header removal
        text = re.sub(r'\bPage\b.*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bDate\b.*', '', text, flags=re.IGNORECASE)

        # Normalize whitespace
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms"""
        technical_keywords = [
            'OS', 'CPU', 'RAM', 'GUI', 'CLI', 'BIOS', 'UEFI', 'GRUB', 'LILO',
            'NTFS', 'ext4', 'TCP', 'IP', 'DNS', 'DHCP', 'Windows', 'Linux',
            'process', 'memory', 'system', 'kernel', 'bootloader', 'security',
            'management', 'file', 'operating', 'interface', 'user', 'boot'
        ]

        found_terms = []
        text_lower = text.lower()

        for term in technical_keywords:
            if term.lower() in text_lower:
                found_terms.append(term)

        return list(set(found_terms))

    def segment_questions(self, text: str) -> Dict[str, str]:
        """Simple question segmentation"""
        questions = {}

        patterns = [
            r'Q\.?\s*\d+',
            r'Question\s*\d+',
            r'\d+\.\s*[A-Z]'
        ]

        for pattern in patterns:
            splits = re.split(pattern, text, flags=re.IGNORECASE)
            if len(splits) > 1:
                for i, segment in enumerate(splits[1:], 1):
                    if segment.strip():
                        questions[f'Q{i}'] = segment.strip()
                return questions

        questions['Q1'] = text
        return questions

    def preprocess_text(self, text: str) -> Dict:
        """Main preprocessing pipeline"""
        result = {
            'original_text': text,
            'student_info': {},
            'questions': {},
            'cleaned_text': '',
            'technical_terms': [],
            'word_count': 0
        }

        # Extract student info
        result['student_info'] = self.extract_student_info(text)

        # Remove headers
        text = self.remove_headers(text)

        # Clean OCR artifacts
        text = self.clean_ocr_artifacts(text)

        # Correct common errors
        text = self.correct_common_errors(text)

        # Deep clean for handwritten/scanned documents
        text = self.deep_clean_ocr_text(text)

        # Normalize text
        text = self.normalize_text(text)

        # Segment questions
        result['questions'] = self.segment_questions(text)

        # Extract technical terms
        result['technical_terms'] = self.extract_technical_terms(text)

        # Final cleaned text
        result['cleaned_text'] = text
        result['word_count'] = len(text.split())

        return result