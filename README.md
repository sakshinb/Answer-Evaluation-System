**🎓 GradeSavvy AI**


GradeSavvy AI is a smart grading assistant for teachers and educators. Upload a student's answer sheet — typed or handwritten — and the system extracts text, analyses it across 5 dimensions, and returns a fair, explainable score with detailed feedback. No manual checking required.
It combines classical NLP techniques (SBERT, NLTK, SymPy) with the Groq LLM (llama-3.3-70b-versatile) to produce grades that are both accurate and interpretable.

✨ Key Features

📄 Multi-format upload — PDF, JPG, PNG, TIFF, WebP, or plain text
✍️ Handwriting OCR — PaddleOCR with TrOCR fallback for scanned/handwritten papers
🏷️ Smart question classification — 6 question types detected automatically
⚖️ Adaptive scoring weights — weights shift per question type (e.g. grammar matters less for code questions)
📊 5 NLP scoring dimensions — semantic, grammar, rubric, factual, completeness
🤖 Groq LLM integration — AI-generated model answers, rubrics, and comparison scoring
🧮 Maths/equation support — SymPy-based symbolic evaluation for numerical answers
📁 CSV export — download full results for record-keeping
🌐 Web interface — clean browser-based UI for uploading and reviewing grades

Architecture:

<img width="590" height="706" alt="image" src="https://github.com/user-attachments/assets/534ef345-f4ae-4133-aafb-aabd5c578f3c" />


🖥️ Demo Screenshots:

<img width="1425" height="624" alt="Screenshot 2026-05-28 164454" src="https://github.com/user-attachments/assets/ba877320-a017-446d-b8fa-199527773b9e" />


👥 Authors
Sakshi Bhingarkar
Vaiahnavi Thorave
Kulshree nakshane
