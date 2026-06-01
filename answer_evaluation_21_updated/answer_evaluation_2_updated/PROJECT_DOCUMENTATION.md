# GradeSavvy AI — Project Documentation

## Overview

GradeSavvy AI is an intelligent answer evaluation platform that grades student handwritten or typed assignment submissions. It combines NLP-based scoring modules with Groq LLM (llama-3.3-70b-versatile) to produce fair, multi-dimensional grades with detailed feedback.

The system supports two grading modes:
- **Standard Grading** — NLP pipeline + Groq LLM blend (50/50)
- **Groq-Enhanced Grading** — Full Groq pipeline with AI-generated model answers and rubrics

---

## Architecture

```
app.py  (Flask API)
  └── grading_system.py  (AdvancedGradingSystem orchestrator)
        ├── modules/semantic_comparator.py
        ├── modules/grammar_checker.py
        ├── modules/rubric_engine.py
        ├── modules/fact_checker.py
        ├── modules/qa_evaluator.py
        ├── modules/scoring_aggregator.py
        ├── modules/groq_grader.py
        ├── modules/gemini_grader.py
        ├── modules/question_type_classifier.py
        ├── modules/adaptive_weight_engine.py
        ├── modules/diagram_ocr_detector.py
        ├── modules/per_question_rubric.py
        ├── modules/question_answer_splitter.py
        ├── modules/list_set_coverage.py
        ├── modules/numerical_sympy_scorer.py
        └── modules/trocr_ocr.py
```

---

## Scoring Pipeline

Each student answer goes through 5 NLP modules. Their scores are weighted and aggregated:

| Module | Default Weight | What it measures |
|---|---|---|
| Semantic | 0.25 | Meaning similarity via SBERT |
| Rubric | 0.10 | Keyword coverage against rubric criteria |
| Grammar | 0.30 | Writing quality and style |
| Factual | 0.15 | Entity and relationship accuracy |
| Completeness | 0.20 | Coverage relative to model answer |

In standard grading, the NLP final score is blended 50/50 with Groq's LLM comparison score.

---
