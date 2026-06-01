import { useState, useEffect } from "react";
import { CheckCircle, XCircle, AlertTriangle, Download, RotateCcw, BookOpen, ListChecks, BarChart2, SplitSquareHorizontal } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import type { GradingResults, StudentResult, PerQuestionResult } from "@/lib/api";
import { exportResults } from "@/lib/api";

interface ResultsSectionProps {
  results: GradingResults | null;
  sessionId: string | null;
  maxMarks: number;
  onReset: () => void;
}

function fmt(score: number, max: number) {
  return `${((score / 100) * max).toFixed(1)}/${max}`;
}

function ScoreBar({ label, score, max }: { label: string; score: number; max: number }) {
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-semibold text-foreground">{fmt(score, max)}</span>
      </div>
      <div className="h-3 bg-muted rounded-full overflow-hidden">
        <div className="h-full bg-gradient-to-r from-primary to-secondary rounded-full transition-all duration-700"
          style={{ width: `${Math.min(score, 100)}%` }} />
      </div>
    </div>
  );
}

function InlineBold({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  if (parts.length === 1) return <>{text}</>;
  return (
    <>
      {parts.map((p, i) =>
        p.startsWith("**")
          ? <strong key={i} className="font-semibold text-foreground">{p.replace(/\*\*/g, "")}</strong>
          : <span key={i}>{p}</span>
      )}
    </>
  );
}

function RenderLine({ line, idx }: { line: string; idx: number }) {
  const h1 = line.match(/^#\s+(.*)/);
  const h2 = line.match(/^##\s+(.*)/);
  const h3 = line.match(/^###\s+(.*)/);
  const h4 = line.match(/^####\s+(.*)/);
  if (h4) return <p key={idx} className="text-sm font-semibold text-muted-foreground mt-2">{h4[1]}</p>;
  if (h3) return <h4 key={idx} className="text-sm font-semibold text-foreground mt-3 mb-1">{h3[1]}</h4>;
  if (h2) return <h3 key={idx} className="text-base font-bold text-foreground mt-4 mb-1">{h2[1]}</h3>;
  if (h1) return <h2 key={idx} className="text-lg font-bold text-primary mt-5 mb-2 border-b border-primary/20 pb-1">{h1[1]}</h2>;
  if (/^[-*]\s/.test(line)) return (
    <div key={idx} className="flex gap-2 text-sm text-muted-foreground ml-2">
      <span className="text-primary shrink-0 mt-0.5">•</span>
      <span className="leading-relaxed"><InlineBold text={line.replace(/^[-*]\s/, "")} /></span>
    </div>
  );
  const num = line.match(/^(\d+)[.)]\s+(.*)/);
  if (num) return (
    <div key={idx} className="flex gap-2 text-sm text-muted-foreground ml-2">
      <span className="text-primary font-semibold shrink-0 w-5">{num[1]}.</span>
      <span className="leading-relaxed"><InlineBold text={num[2]} /></span>
    </div>
  );
  if (!line.trim()) return <div key={idx} className="h-2" />;
  return <p key={idx} className="text-sm text-muted-foreground leading-relaxed"><InlineBold text={line} /></p>;
}

function ModelAnswerView({ text }: { text: string }) {
  if (!text) return <p className="text-sm text-muted-foreground italic">No model answer available.</p>;
  const lines = text.split("\n");
  const qPattern = /^(#{1,4}\s*)?(Question\s+\d+|Q\s*\d+)\s*[:.)]/i;
  const sections: { heading: string; body: string[] }[] = [];
  let cur: { heading: string; body: string[] } | null = null;
  for (const line of lines) {
    if (qPattern.test(line.trim())) {
      if (cur) sections.push(cur);
      cur = { heading: line.trim().replace(/^#+\s*/, "").replace(/\*\*/g, ""), body: [] };
    } else {
      if (cur) cur.body.push(line);
      else { if (!sections.length) sections.push({ heading: "", body: [] }); sections[0].body.push(line); }
    }
  }
  if (cur) sections.push(cur);
  if (sections.length === 1 && !sections[0].heading) {
    return <div className="space-y-1">{lines.map((l, i) => <RenderLine key={i} line={l} idx={i} />)}</div>;
  }
  return (
    <div className="space-y-6">
      {sections.map((sec, si) => {
        if (!sec.heading && sec.body.every(l => !l.trim())) return null;
        return (
          <div key={si} className="rounded-xl overflow-hidden border border-primary/25 shadow-soft">
            {sec.heading && (
              <div className="bg-gradient-to-r from-primary/10 to-secondary/10 border-b border-primary/20 px-6 py-4 flex items-start gap-3">
                <span className="w-7 h-7 rounded-full bg-primary text-primary-foreground text-xs font-bold flex items-center justify-center shrink-0 mt-0.5">{si}</span>
                <span className="font-bold text-base text-foreground leading-snug">{sec.heading}</span>
              </div>
            )}
            <div className="px-6 py-5 space-y-1 bg-card/60">
              {sec.body.map((l, li) => <RenderLine key={li} line={l} idx={li} />)}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function RubricView({ rubric }: { rubric: StudentResult["gemini_rubric"] }) {
  if (!rubric?.criteria?.length) return <p className="text-sm text-muted-foreground">No rubric data available.</p>;
  return (
    <div className="space-y-4">
      {rubric.general_instructions && (
        <p className="text-sm text-muted-foreground italic border-l-4 border-primary/30 pl-4 py-1 bg-primary/5 rounded-r-lg">{rubric.general_instructions}</p>
      )}
      <div className="grid md:grid-cols-2 gap-4">
        {rubric.criteria.map((c, i) => (
          <div key={i} className="border border-border/60 rounded-xl p-4 space-y-2 bg-card/60">
            <div className="flex items-center justify-between">
              <span className="font-bold text-sm">{c.name}</span>
              <span className="text-xs bg-primary/10 text-primary px-2.5 py-1 rounded-full font-semibold">{c.max_marks} marks</span>
            </div>
            <p className="text-xs text-muted-foreground">{c.description}</p>
            {c.keywords?.length > 0 && (
              <div className="flex flex-wrap gap-1">{c.keywords.map((k, j) => <span key={j} className="text-[10px] bg-muted px-2 py-0.5 rounded-full">{k}</span>)}</div>
            )}
            <div className="grid grid-cols-2 gap-2 pt-1">
              {Object.entries(c.performance_levels).map(([level, desc]) => (
                <div key={level} className="text-[11px] bg-muted/50 rounded-lg p-2">
                  <span className="font-semibold capitalize text-foreground block mb-0.5">{level}</span>
                  <span className="text-muted-foreground">{String(desc)}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Question type badge colours ────────────────────────────────────────────────
const Q_TYPE_META: Record<string, { label: string; color: string }> = {
  factual:    { label: "Factual",     color: "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300" },
  conceptual: { label: "Conceptual",  color: "bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300" },
  analytical: { label: "Analytical",  color: "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300" },
  numerical:  { label: "Numerical",   color: "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300" },
  code:       { label: "Code",        color: "bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-300" },
  list:       { label: "List",        color: "bg-teal-100 text-teal-700 dark:bg-teal-900/40 dark:text-teal-300" },
};

// Friendly labels for routed_score_detail keys
const DETAIL_LABELS: Record<string, string> = {
  concept_coverage_pct: "Concept coverage",
  depth_pct: "Depth",
  has_examples: "Uses examples",
  has_definitions: "Has definitions",
  has_transitions: "Transition words",
  kw_coverage_pct: "Keyword coverage",
  has_arguments: "Makes arguments",
  has_counterpoints: "Counterpoints present",
  has_evidence: "Cites evidence",
  correct_entities: "Correct entities",
  missing_entities: "Missing entities",
  contradiction_penalty: "Contradiction penalty",
  matched: "Matched values",
  total_ref: "Expected values",
  final_answer_bonus: "Final answer correct",
  ast_node_similarity_pct: "AST node similarity",
  identifier_similarity_pct: "Identifier similarity",
  syntax_valid: "Valid syntax",
  matched_items: "Matched items",
  coverage_pct: "List coverage",
};

function formatDetailValue(val: unknown): string {
  if (typeof val === "boolean") return val ? "✓ Yes" : "✗ No";
  if (typeof val === "number") return `${val}${String(val).includes(".") ? "" : ""}`;
  if (Array.isArray(val)) return val.slice(0, 3).join(", ") || "—";
  return String(val ?? "—");
}

function RoutedScoreDetail({ detail, qType }: { detail: Record<string, unknown>; qType: string }) {
  const entries = Object.entries(detail).filter(
    ([k]) => !["method", "groq", "error", "ref_lines", "student_lines", "ref_item_count",
                "student_item_count", "total_ref_entities", "total_ref_numbers",
                "matched_examples", "match", "ratio"].includes(k)
  );
  if (!entries.length) return null;
  return (
    <div className="mt-3">
      <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
        {Q_TYPE_META[qType]?.label ?? qType} scorer detail
      </p>
      <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs text-muted-foreground">
        {entries.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-2">
            <span className="truncate">{DETAIL_LABELS[k] ?? k}</span>
            <span className="font-semibold text-foreground shrink-0">{formatDetailValue(v)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── NEW: per-question breakdown component ──────────────────────────────────────
function BreakdownView({
  perQuestionResults,
  maxMarks,
}: {
  perQuestionResults: PerQuestionResult[];
  maxMarks: number;
}) {
  const [openIdx, setOpenIdx] = useState<number | null>(0);

  if (!perQuestionResults?.length) {
    return (
      <p className="text-sm text-muted-foreground italic">
        No per-question breakdown available. This may be a single-question submission.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      {perQuestionResults.map((pq, i) => {
        const isOpen = openIdx === i;
        const score = pq.score ?? null;
        const scoreColor =
          score === null ? "text-muted-foreground"
          : score >= 70 ? "text-secondary"
          : score >= 50 ? "text-accent"
          : "text-destructive";

        return (
          <div
            key={i}
            className="border border-border/60 rounded-xl overflow-hidden bg-card/60"
          >
            {/* Header — always visible, click to expand */}
            <button
              className="w-full flex items-center justify-between px-5 py-4 text-left hover:bg-muted/30 transition-colors"
              onClick={() => setOpenIdx(isOpen ? null : i)}
            >
              <div className="flex items-center gap-3 min-w-0">
                <span className="w-7 h-7 rounded-full bg-primary/10 text-primary text-xs font-bold flex items-center justify-center shrink-0">
                  Q{pq.question_num}
                </span>
                <span className="text-sm font-semibold text-foreground truncate">
                  {pq.question}
                </span>
                {pq.question_type && (
                  <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full shrink-0 ${Q_TYPE_META[pq.question_type]?.color ?? "bg-muted text-muted-foreground"}`}>
                    {Q_TYPE_META[pq.question_type]?.label ?? pq.question_type}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-3 shrink-0 ml-4">
                {score !== null && (
                  <span className={`text-base font-bold ${scoreColor}`}>
                    {fmt(score, maxMarks)}
                  </span>
                )}
                <span className="text-muted-foreground text-lg">{isOpen ? "▲" : "▼"}</span>
              </div>
            </button>

            {/* Expanded detail */}
            {isOpen && (
              <div className="px-5 pb-5 space-y-4 border-t border-border/40">

                {/* Student's aligned answer excerpt */}
                {pq.student_answer_excerpt && (
                  <div className="mt-4">
                    <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1.5">
                      Student's answer (excerpt)
                    </p>
                    <div className="bg-muted/40 rounded-lg px-4 py-3 text-sm text-muted-foreground italic leading-relaxed">
                      {pq.student_answer_excerpt}
                      {pq.student_answer_excerpt.length >= 300 && (
                        <span className="not-italic text-primary/60"> …</span>
                      )}
                    </div>
                  </div>
                )}

                {/* Per-question score bars (only in standard mode) */}
                {pq.semantic_score !== undefined && (
                  <div className="grid md:grid-cols-2 gap-x-10 gap-y-3 pt-1">
                    <ScoreBar label="Type-aware score" score={pq.semantic_score ?? 0} max={maxMarks} />
                    <ScoreBar label="Rubric"       score={pq.rubric_score ?? 0}       max={maxMarks} />
                    <ScoreBar label="Grammar"      score={pq.grammar_score ?? 0}      max={maxMarks} />
                    <ScoreBar label="Factual"      score={pq.factual_score ?? 0}      max={maxMarks} />
                    <ScoreBar label="Completeness" score={pq.completeness_score ?? 0} max={maxMarks} />
                  </div>
                )}

                {/* Routed scorer breakdown */}
                {pq.routed_score_detail && pq.question_type && (
                  <RoutedScoreDetail
                    detail={pq.routed_score_detail as Record<string, unknown>}
                    qType={pq.question_type}
                  />
                )}

                {pq.used_groq_fallback && (
                  <p className="text-[10px] text-muted-foreground italic mt-1">
                    ⚡ Groq fallback was used for this question
                  </p>
                )}

                {/* Per-question feedback */}
                {(pq.strengths?.length || pq.weaknesses?.length || pq.suggestions?.length) && (
                  <div className="grid md:grid-cols-3 gap-4 text-sm">
                    {pq.strengths?.length > 0 && (
                      <div className="bg-secondary/5 border border-secondary/20 rounded-xl p-3">
                        <p className="font-semibold flex items-center gap-1 text-secondary mb-1.5">
                          <CheckCircle className="w-3.5 h-3.5" /> Strengths
                        </p>
                        <ul className="pl-3 list-disc text-muted-foreground space-y-0.5 text-xs">
                          {pq.strengths.map((s, j) => <li key={j}>{s}</li>)}
                        </ul>
                      </div>
                    )}
                    {pq.weaknesses?.length > 0 && (
                      <div className="bg-destructive/5 border border-destructive/20 rounded-xl p-3">
                        <p className="font-semibold flex items-center gap-1 text-destructive mb-1.5">
                          <XCircle className="w-3.5 h-3.5" /> Weaknesses
                        </p>
                        <ul className="pl-3 list-disc text-muted-foreground space-y-0.5 text-xs">
                          {pq.weaknesses.map((w, j) => <li key={j}>{w}</li>)}
                        </ul>
                      </div>
                    )}
                    {pq.suggestions?.length > 0 && (
                      <div className="bg-accent/5 border border-accent/20 rounded-xl p-3">
                        <p className="font-semibold flex items-center gap-1 text-accent mb-1.5">
                          <AlertTriangle className="w-3.5 h-3.5" /> Suggestions
                        </p>
                        <ul className="pl-3 list-disc text-muted-foreground space-y-0.5 text-xs">
                          {pq.suggestions.map((s, j) => <li key={j}>{s}</li>)}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function StudentCard({ result, maxMarks }: { result: StudentResult; maxMarks: number }) {
  const hasBreakdown = (result.per_question_results?.length ?? 0) > 0;
  const [tab, setTab] = useState<"scores" | "breakdown" | "model" | "rubric">("scores");
  const score = result.final_score ?? 0;
  const displayScore = ((score / 100) * maxMarks).toFixed(1);
  const color = score >= 70 ? "text-secondary" : score >= 50 ? "text-accent" : "text-destructive";
  const name = result.student_info?.name || result.filename;
  const modelAnswer = result.gemini_model_answer || (result.model_answer_used as string) || "";
  const rubric = result.gemini_rubric || (result.rubric_used as StudentResult["gemini_rubric"]) || null;

  const tabs = [
    { key: "scores"    as const, label: "Scores",       icon: <BarChart2 className="w-3.5 h-3.5" /> },
    ...(hasBreakdown ? [{ key: "breakdown" as const, label: "By Question", icon: <SplitSquareHorizontal className="w-3.5 h-3.5" /> }] : []),
    { key: "model"     as const, label: "Model Answer", icon: <BookOpen className="w-3.5 h-3.5" /> },
    { key: "rubric"    as const, label: "Rubric",       icon: <ListChecks className="w-3.5 h-3.5" /> },
  ];

  return (
    <Card className="w-full p-8 bg-card/90 border-border/50 shadow-medium">
      <div className="flex items-center justify-between mb-6">
        <div className="flex-1 min-w-0 mr-4">
          <p className="font-bold text-xl truncate">{name}</p>
          <p className="text-sm text-muted-foreground truncate mt-0.5">{result.filename}</p>
        </div>
        <div className={`text-5xl font-bold shrink-0 ${color}`}>
          {displayScore}<span className="text-xl text-muted-foreground">/{maxMarks}</span>
        </div>
      </div>

      <div className="flex gap-1.5 mb-6 bg-muted/40 rounded-xl p-1.5">
        {tabs.map(t => (
          <button key={t.key} onClick={() => setTab(t.key)}
            className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-semibold transition-all
              ${tab === t.key ? "bg-background shadow text-primary" : "text-muted-foreground hover:text-foreground"}`}>
            {t.icon}{t.label}
          </button>
        ))}
      </div>

      {tab === "scores" && (
        <div className="space-y-5">
          <div className="grid md:grid-cols-2 gap-x-12 gap-y-4">
            <ScoreBar label="Semantic"     score={result.semantic_score ?? 0}     max={maxMarks} />
            <ScoreBar label="Rubric"       score={result.rubric_score ?? 0}       max={maxMarks} />
            <ScoreBar label="Grammar"      score={result.grammar_score ?? 0}      max={maxMarks} />
            <ScoreBar label="Factual"      score={result.factual_score ?? 0}      max={maxMarks} />
            <ScoreBar label="Completeness" score={result.completeness_score ?? 0} max={maxMarks} />
          </div>
          <div className="pt-2 grid md:grid-cols-3 gap-5 text-sm">
            {result.strengths?.length > 0 && (
              <div className="bg-secondary/5 border border-secondary/20 rounded-xl p-4">
                <p className="font-semibold flex items-center gap-1.5 text-secondary mb-2"><CheckCircle className="w-4 h-4" /> Strengths</p>
                <ul className="pl-4 list-disc text-muted-foreground space-y-1">{result.strengths.map((s, i) => <li key={i}>{s}</li>)}</ul>
              </div>
            )}
            {result.weaknesses?.length > 0 && (
              <div className="bg-destructive/5 border border-destructive/20 rounded-xl p-4">
                <p className="font-semibold flex items-center gap-1.5 text-destructive mb-2"><XCircle className="w-4 h-4" /> Weaknesses</p>
                <ul className="pl-4 list-disc text-muted-foreground space-y-1">{result.weaknesses.map((w, i) => <li key={i}>{w}</li>)}</ul>
              </div>
            )}
            {result.suggestions?.length > 0 && (
              <div className="bg-accent/5 border border-accent/20 rounded-xl p-4">
                <p className="font-semibold flex items-center gap-1.5 text-accent mb-2"><AlertTriangle className="w-4 h-4" /> Suggestions</p>
                <ul className="pl-4 list-disc text-muted-foreground space-y-1">{result.suggestions.map((s, i) => <li key={i}>{s}</li>)}</ul>
              </div>
            )}
          </div>
        </div>
      )}

      {tab === "breakdown" && (
        <BreakdownView
          perQuestionResults={result.per_question_results ?? []}
          maxMarks={maxMarks}
        />
      )}

      {tab === "model" && <ModelAnswerView text={modelAnswer} />}
      {tab === "rubric" && <RubricView rubric={rubric} />}
    </Card>
  );
}

export function ResultsSection({ results, sessionId, maxMarks, onReset }: ResultsSectionProps) {
  const [animatedAvg, setAnimatedAvg] = useState(0);
  useEffect(() => {
    if (!results) return;
    const target = (results.summary.average_score / 100) * maxMarks;
    let cur = 0;
    const inc = target / 50;
    const t = setInterval(() => {
      cur += inc;
      if (cur >= target) { setAnimatedAvg(target); clearInterval(t); }
      else setAnimatedAvg(Math.round(cur * 10) / 10);
    }, 30);
    return () => clearInterval(t);
  }, [results, maxMarks]);

  if (!results) return null;
  const { summary } = results;
  const successful = results.results.filter(r => !r.processing_failed);
  const failed     = results.results.filter(r => r.processing_failed);
  const hi = ((summary.highest_score / 100) * maxMarks).toFixed(1);
  const lo = ((summary.lowest_score  / 100) * maxMarks).toFixed(1);

  return (
    <section className="py-10 px-4 md:px-10 bg-gradient-hero min-h-screen">
      <div className="w-full max-w-screen-xl mx-auto">
        <div className="text-center mb-8">
          <h2 className="text-3xl md:text-4xl font-bold mb-1 bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">Grading Results</h2>
          <p className="text-muted-foreground text-sm">{summary.successful_gradings} of {summary.total_files} files graded · scored out of {maxMarks}</p>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          {[
            { label: "Average Score", value: `${animatedAvg.toFixed(1)}/${maxMarks}` },
            { label: "Highest Score", value: `${hi}/${maxMarks}` },
            { label: "Lowest Score",  value: `${lo}/${maxMarks}` },
            { label: "Files Graded",  value: `${summary.successful_gradings}` },
          ].map((m, i) => (
            <Card key={i} className="p-5 text-center bg-card/90 border-border/50">
              <p className="text-3xl font-bold text-primary">{m.value}</p>
              <p className="text-sm text-muted-foreground mt-1">{m.label}</p>
            </Card>
          ))}
        </div>
        <div className="flex flex-col gap-6 mb-8">
          {successful.map((r, i) => <StudentCard key={i} result={r} maxMarks={maxMarks} />)}
        </div>
        {failed.length > 0 && (
          <Card className="p-4 mb-8 border-destructive/30 bg-destructive/5">
            <p className="font-medium text-destructive mb-2">Failed: {failed.length} file(s)</p>
            {failed.map((r, i) => <p key={i} className="text-sm text-muted-foreground">{r.filename}: {r.error}</p>)}
          </Card>
        )}
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Button variant="hero" size="lg" onClick={onReset}><RotateCcw className="w-4 h-4 mr-2" /> Grade Again</Button>
          {sessionId && (
            <Button variant="outline" size="lg" onClick={() => exportResults(sessionId)}><Download className="w-4 h-4 mr-2" /> Export CSV</Button>
          )}
        </div>
      </div>
    </section>
  );
}
