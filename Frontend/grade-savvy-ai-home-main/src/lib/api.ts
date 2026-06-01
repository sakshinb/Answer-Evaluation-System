const BASE = '/api';

export async function newSession(): Promise<string> {
  const res = await fetch(`${BASE}/new_session`, { method: 'POST' });
  const data = await res.json();
  if (!data.success) throw new Error(data.message);
  return data.session_id;
}

export async function uploadFile(sessionId: string, file: File): Promise<void> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/upload_file/${sessionId}`, { method: 'POST', body: form });
  const data = await res.json();
  if (!data.success) throw new Error(data.message);
}

export async function setQuestions(sessionId: string, questions: string[]): Promise<void> {
  const res = await fetch(`${BASE}/set_questions/${sessionId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ questions }),
  });
  const data = await res.json();
  if (!data.success) throw new Error(data.message);
}

export async function setModelAnswer(sessionId: string, modelAnswer: string): Promise<void> {
  const res = await fetch(`${BASE}/set_model_answer/${sessionId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_answer: modelAnswer }),
  });
  const data = await res.json();
  if (!data.success) throw new Error(data.message);
}

export async function setRubricManual(sessionId: string, rubric: object): Promise<void> {
  const res = await fetch(`${BASE}/set_rubric_manual/${sessionId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rubric }),
  });
  const data = await res.json();
  if (!data.success) throw new Error(data.message);
}

export async function startGrading(sessionId: string): Promise<void> {
  const res = await fetch(`${BASE}/start_grading/${sessionId}`, { method: 'POST' });
  const data = await res.json();
  if (!data.success) throw new Error(data.message);
}

export async function startGeminiGrading(sessionId: string): Promise<void> {
  const res = await fetch(`${BASE}/start_gemini_grading/${sessionId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  });
  const data = await res.json();
  if (!data.success) throw new Error(data.message);
}

export async function getResults(sessionId: string): Promise<GradingResults> {
  const res = await fetch(`${BASE}/get_results/${sessionId}`);
  const data = await res.json();
  if (!data.success) throw new Error(data.message);
  return data;
}

export async function exportResults(sessionId: string): Promise<void> {
  window.open(`${BASE}/export_results/${sessionId}`, '_blank');
}

export async function sendChatMessage(
  message: string,
  history: Array<{ role: string; content: string }>
): Promise<string> {
  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history }),
  });
  const data = await res.json();
  if (!data.success) throw new Error(data.message);
  return data.reply;
}

export interface StudentResult {
  filename: string;
  student_id: string;
  final_score: number;        // 0-100 scale from backend
  semantic_score: number;
  rubric_score: number;
  grammar_score: number;
  factual_score: number;
  completeness_score: number;
  confidence_score: number;
  word_count?: number;
  sentence_count?: number;
  strengths: string[];
  weaknesses: string[];
  suggestions: string[];
  student_info?: { name?: string };
  processing_failed?: boolean;
  error?: string;
  gemini_used?: boolean;
  // model answer & rubric (present in both modes)
  model_answer_used?: string;
  rubric_used?: Record<string, unknown>;
  gemini_model_answer?: string;
  gemini_rubric?: {
    criteria?: Array<{
      name: string;
      description: string;
      max_marks: number;
      keywords: string[];
      performance_levels: Record<string, string>;
    }>;
    total_marks?: number;
    general_instructions?: string;
  };
}

export interface GradingResults {
  results: StudentResult[];
  summary: {
    total_files: number;
    successful_gradings: number;
    average_score: number;
    highest_score: number;
    lowest_score: number;
  };
  session_info: {
    session_id: string;
    created_at: string;
    questions_count: number;
    files_count: number;
  };
}

