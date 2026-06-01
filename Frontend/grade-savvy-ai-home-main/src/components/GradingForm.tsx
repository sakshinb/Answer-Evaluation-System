import { useState, useRef } from "react";
import { Upload, Sparkles, X, Plus, Trash2, ChevronRight, ChevronLeft, Cpu, Zap, FileText, ListChecks } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Stepper } from "@/components/Stepper";
import { useToast } from "@/hooks/use-toast";

export interface GradingPayload {
  questions: string[];
  files: File[];
  mode: "standard" | "groq";
  maxMarks: number;
  modelAnswer?: string;
  rubricCriteria?: Record<string, string>;
}

interface GradingFormProps {
  onSubmit: (payload: GradingPayload) => Promise<void>;
  isLoading: boolean;
}

// Groq: Model → Upload → Questions → Marks
const GROQ_STEPS = [
  { id: 1, label: "Model",     description: "Choose AI mode"  },
  { id: 2, label: "Upload",    description: "Student files"   },
  { id: 3, label: "Questions", description: "Enter questions" },
  { id: 4, label: "Marks",     description: "Set max marks"   },
];

// Standard: Model → Upload → Questions → Model Answer → Rubric → Marks
const STD_STEPS = [
  { id: 1, label: "Model",         description: "Choose AI mode"    },
  { id: 2, label: "Upload",        description: "Student files"     },
  { id: 3, label: "Questions",     description: "Enter questions"   },
  { id: 4, label: "Model Answer",  description: "Reference answer"  },
  { id: 5, label: "Rubric",        description: "Grading criteria"  },
  { id: 6, label: "Marks",         description: "Set max marks"     },
];

export function GradingForm({ onSubmit, isLoading }: GradingFormProps) {
  const [step, setStep]           = useState(1);
  const [mode, setMode]           = useState<"standard" | "groq">("groq");
  const [files, setFiles]         = useState<File[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [questions, setQuestions] = useState<string[]>([""]);
  const [modelAnswers, setModelAnswers] = useState<string[]>([""]);
  const [rubricRows, setRubricRows]   = useState<{ name: string; desc: string }[]>([
    { name: "Content Accuracy",       desc: "Correctness of facts and concepts" },
    { name: "Completeness",           desc: "Coverage of all required topics"   },
    { name: "Clarity & Organization", desc: "Clear and logical structure"       },
  ]);
  const [maxMarks, setMaxMarks]   = useState(10);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const steps = mode === "standard" ? STD_STEPS : GROQ_STEPS;
  const lastStep = steps.length;

  /* ── file helpers ── */
  const handleFiles = (incoming: FileList | null) => {
    if (!incoming) return;
    const valid = Array.from(incoming).filter(f => {
      const ext = f.name.split(".").pop()?.toLowerCase();
      if (!["pdf", "txt"].includes(ext ?? "")) {
        toast({ title: "Invalid file", description: `${f.name} must be PDF or TXT`, variant: "destructive" });
        return false;
      }
      return true;
    });
    setFiles(prev => [...prev, ...valid]);
  };

  /* ── question helpers ── */
  const addQ    = () => { setQuestions(q => [...q, ""]); setModelAnswers(a => [...a, ""]); };
  const removeQ = (i: number) => { setQuestions(q => q.filter((_, idx) => idx !== i)); setModelAnswers(a => a.filter((_, idx) => idx !== i)); };
  const updateQ = (i: number, v: string) => setQuestions(q => q.map((x, idx) => idx === i ? v : x));

  /* ── rubric helpers ── */
  const addRow    = () => setRubricRows(r => [...r, { name: "", desc: "" }]);
  const removeRow = (i: number) => setRubricRows(r => r.filter((_, idx) => idx !== i));
  const updateRow = (i: number, field: "name" | "desc", v: string) =>
    setRubricRows(r => r.map((row, idx) => idx === i ? { ...row, [field]: v } : row));

  /* ── navigation ── */
  const canNext = (): boolean => {
    if (step === 1) return true;
    if (step === 2) return files.length > 0;
    if (step === 3) return questions.some(q => q.trim());
    if (mode === "standard") {
      if (step === 4) return modelAnswers.some(a => a.trim().length > 0);
      if (step === 5) return rubricRows.some(r => r.name.trim());
      if (step === 6) return maxMarks > 0;
    } else {
      if (step === 4) return maxMarks > 0;
    }
    return true;
  };

  const next = () => {
    if (!canNext()) {
      const msgs: Record<number, string> = {
        2: "Upload at least one file",
        3: "Add at least one question",
        4: mode === "standard" ? "Enter a model answer" : "Max marks must be > 0",
        5: "Add at least one rubric criterion",
        6: "Max marks must be > 0",
      };
      toast({ title: "Required", description: msgs[step] ?? "", variant: "destructive" });
      return;
    }
    setStep(s => s + 1);
  };

  const back = () => setStep(s => s - 1);

  const handleGrade = async () => {
    const rubricCriteria = mode === "standard"
      ? Object.fromEntries(rubricRows.filter(r => r.name.trim()).map(r => [r.name, r.desc]))
      : undefined;
    await onSubmit({
      questions: questions.filter(q => q.trim()),
      files,
      mode,
      maxMarks,
      modelAnswer: mode === "standard" ? modelAnswers.map((a, i) => `Question ${i + 1}: ${questions[i] || ""}\n${a}`).join("\n\n") : undefined,
      rubricCriteria,
    });
  };

  // When mode changes, reset to step 1
  const switchMode = (m: "standard" | "groq") => {
    setMode(m);
    setStep(1);
  };

  return (
    <div className="min-h-screen bg-gradient-hero py-8 px-4">
      <div className="container mx-auto max-w-3xl">

        <div className="text-center mb-6">
          <h2 className="text-3xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
            Grade Answers
          </h2>
          <p className="text-muted-foreground mt-1 text-sm">Follow the steps to set up and run grading</p>
        </div>

        <Stepper steps={steps} current={step} />

        <Card className="p-8 mt-4 shadow-medium bg-card/90 backdrop-blur-sm border-border/50 min-h-72">

          {/* STEP 1 — Mode */}
          {step === 1 && (
            <div className="space-y-4">
              <h3 className="text-xl font-semibold">Choose Grading Mode</h3>
              <p className="text-sm text-muted-foreground">Select how you want answers to be evaluated.</p>
              <div className="grid sm:grid-cols-2 gap-4 mt-4">
                {([
                  { key: "standard" as const, icon: <Cpu className="w-8 h-8 text-primary" />,    title: "Standard NLP",      desc: "You provide the model answer and rubric. Fast local NLP scoring." },
                  { key: "groq"     as const, icon: <Zap className="w-8 h-8 text-secondary" />,  title: "Groq AI Enhanced",  desc: "Groq auto-generates model answer and rubric using llama-3.3-70b." },
                ]).map(({ key, icon, title, desc }) => (
                  <button key={key} type="button" onClick={() => switchMode(key)}
                    className={`text-left p-5 rounded-xl border-2 transition-all duration-200 space-y-2
                      ${mode === key ? "border-primary bg-primary/8 shadow-soft" : "border-border hover:border-primary/40"}`}>
                    <div className="flex items-center gap-3">
                      {icon}
                      <span className="font-semibold">{title}</span>
                      {mode === key && <span className="ml-auto text-xs bg-primary text-primary-foreground px-2 py-0.5 rounded-full">Selected</span>}
                    </div>
                    <p className="text-xs text-muted-foreground leading-relaxed">{desc}</p>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* STEP 2 — Upload */}
          {step === 2 && (
            <div className="space-y-4">
              <h3 className="text-xl font-semibold">Upload Student Answer Files</h3>
              <p className="text-sm text-muted-foreground">PDF or TXT files, up to 16 MB each.</p>
              <div
                className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all duration-200
                  ${dragActive ? "border-primary bg-primary/5" : "border-border hover:border-primary hover:bg-primary/5"}`}
                onDrop={e => { e.preventDefault(); setDragActive(false); handleFiles(e.dataTransfer.files); }}
                onDragOver={e => { e.preventDefault(); setDragActive(true); }}
                onDragLeave={() => setDragActive(false)}
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload className="w-10 h-10 mx-auto mb-3 text-muted-foreground" />
                <p className="font-medium">Drop files here or click to browse</p>
                <p className="text-xs text-muted-foreground mt-1">PDF · TXT</p>
              </div>
              <input ref={fileInputRef} type="file" accept=".pdf,.txt" multiple className="hidden"
                onChange={e => handleFiles(e.target.files)} />
              {files.length > 0 && (
                <ul className="space-y-2 max-h-48 overflow-y-auto">
                  {files.map((f, i) => (
                    <li key={i} className="flex items-center justify-between bg-muted/30 rounded-lg px-3 py-2 text-sm">
                      <span className="truncate max-w-xs">{f.name}</span>
                      <div className="flex items-center gap-2 text-muted-foreground shrink-0">
                        <span>{(f.size / 1024).toFixed(0)} KB</span>
                        <button type="button" onClick={() => setFiles(fs => fs.filter((_, idx) => idx !== i))}>
                          <X className="w-4 h-4 hover:text-destructive" />
                        </button>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}

          {/* STEP 3 — Questions */}
          {step === 3 && (
            <div className="space-y-4">
              <h3 className="text-xl font-semibold">Enter Questions</h3>
              <p className="text-sm text-muted-foreground">Add the questions students were asked to answer.</p>
              <div className="space-y-3 max-h-80 overflow-y-auto pr-1">
                {questions.map((q, i) => (
                  <div key={i} className="flex gap-2 items-start">
                    <span className="mt-2.5 text-xs font-bold text-muted-foreground w-5 shrink-0">{i + 1}.</span>
                    <Textarea placeholder={`Question ${i + 1}...`} value={q}
                      onChange={e => updateQ(i, e.target.value)} className="min-h-16 text-sm resize-none flex-1" />
                    {questions.length > 1 && (
                      <button type="button" onClick={() => removeQ(i)} className="mt-2">
                        <Trash2 className="w-4 h-4 text-muted-foreground hover:text-destructive" />
                      </button>
                    )}
                  </div>
                ))}
              </div>
              <Button type="button" variant="outline" size="sm" onClick={addQ}>
                <Plus className="w-4 h-4 mr-1" /> Add Question
              </Button>
            </div>
          )}

          {/* STEP 4 (standard) — Model Answer */}
          {step === 4 && mode === "standard" && (
            <div className="space-y-4">
              <h3 className="text-xl font-semibold flex items-center gap-2">
                <FileText className="w-5 h-5 text-primary" /> Model Answer
              </h3>
              <p className="text-sm text-muted-foreground">
                Provide the ideal answer for each question. Students will be scored against these.
              </p>
              <div className="space-y-5 max-h-[28rem] overflow-y-auto pr-1">
                {questions.filter(q => q.trim()).map((q, i) => (
                  <div key={i} className="space-y-2">
                    <div className="flex items-start gap-2">
                      <span className="w-6 h-6 rounded-full bg-primary text-primary-foreground text-xs font-bold flex items-center justify-center shrink-0 mt-0.5">
                        {i + 1}
                      </span>
                      <p className="text-sm font-medium text-foreground leading-snug">{q}</p>
                    </div>
                    <Textarea
                      placeholder={`Model answer for question ${i + 1}...`}
                      value={modelAnswers[i] ?? ""}
                      onChange={e => setModelAnswers(a => a.map((v, idx) => idx === i ? e.target.value : v))}
                      className="min-h-28 text-sm resize-y ml-8"
                    />
                    <p className="text-xs text-muted-foreground ml-8">
                      {(modelAnswers[i] ?? "").trim().split(/\s+/).filter(Boolean).length} words
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* STEP 5 (standard) — Rubric */}
          {step === 5 && mode === "standard" && (
            <div className="space-y-4">
              <h3 className="text-xl font-semibold flex items-center gap-2">
                <ListChecks className="w-5 h-5 text-primary" /> Grading Rubric
              </h3>
              <p className="text-sm text-muted-foreground">
                Define the criteria used to evaluate student answers.
              </p>
              <div className="space-y-3 max-h-72 overflow-y-auto pr-1">
                {rubricRows.map((row, i) => (
                  <div key={i} className="flex gap-2 items-start bg-muted/20 rounded-lg p-3">
                    <div className="flex-1 space-y-2">
                      <Input placeholder="Criterion name (e.g. Content Accuracy)"
                        value={row.name} onChange={e => updateRow(i, "name", e.target.value)}
                        className="text-sm font-medium" />
                      <Input placeholder="Description (e.g. Correctness of facts)"
                        value={row.desc} onChange={e => updateRow(i, "desc", e.target.value)}
                        className="text-sm text-muted-foreground" />
                    </div>
                    {rubricRows.length > 1 && (
                      <button type="button" onClick={() => removeRow(i)} className="mt-1 shrink-0">
                        <Trash2 className="w-4 h-4 text-muted-foreground hover:text-destructive" />
                      </button>
                    )}
                  </div>
                ))}
              </div>
              <Button type="button" variant="outline" size="sm" onClick={addRow}>
                <Plus className="w-4 h-4 mr-1" /> Add Criterion
              </Button>
            </div>
          )}

          {/* STEP 4 (groq) or STEP 6 (standard) — Marks */}
          {((step === 4 && mode === "groq") || (step === 6 && mode === "standard")) && (
            <div className="space-y-6">
              <h3 className="text-xl font-semibold">Set Maximum Marks</h3>
              <p className="text-sm text-muted-foreground">Define the total marks each answer is graded out of.</p>
              <div className="flex items-center gap-6 mt-4">
                <div className="space-y-2 flex-1">
                  <Label htmlFor="maxMarks" className="font-medium">Maximum Marks</Label>
                  <Input id="maxMarks" type="number" min={1} max={100} value={maxMarks}
                    onChange={e => setMaxMarks(Number(e.target.value))}
                    className="text-lg font-semibold w-32" />
                </div>
                <div className="text-center p-6 rounded-xl bg-primary/10 border border-primary/20">
                  <p className="text-5xl font-bold text-primary">{maxMarks}</p>
                  <p className="text-xs text-muted-foreground mt-1">Total Marks</p>
                </div>
              </div>
              <div className="p-4 bg-muted/30 rounded-lg text-sm text-muted-foreground space-y-1">
                <p className="font-medium text-foreground">Summary</p>
                <p>Mode: <span className="font-medium text-foreground">{mode === "groq" ? "Groq AI Enhanced" : "Standard NLP"}</span></p>
                <p>Files: <span className="font-medium text-foreground">{files.length} uploaded</span></p>
                <p>Questions: <span className="font-medium text-foreground">{questions.filter(q => q.trim()).length}</span></p>
                {mode === "standard" && <p>Model answer: <span className="font-medium text-foreground">{modelAnswers.some(a => a.trim()) ? "✓ Provided" : "—"}</span></p>}
                {mode === "standard" && <p>Rubric criteria: <span className="font-medium text-foreground">{rubricRows.filter(r => r.name.trim()).length}</span></p>}
                <p>Max Marks: <span className="font-medium text-foreground">{maxMarks}</span></p>
              </div>
            </div>
          )}

        </Card>

        {/* Navigation */}
        <div className="flex justify-between mt-6">
          <Button variant="outline" onClick={back} disabled={step === 1 || isLoading}>
            <ChevronLeft className="w-4 h-4 mr-1" /> Back
          </Button>
          {step < lastStep ? (
            <Button variant="hero" onClick={next}>
              Next <ChevronRight className="w-4 h-4 ml-1" />
            </Button>
          ) : (
            <Button variant="hero" onClick={handleGrade} disabled={isLoading} className="min-w-40">
              {isLoading
                ? <><div className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin mr-2" />Grading...</>
                : <><Sparkles className="w-4 h-4 mr-2" />Start Grading</>}
            </Button>
          )}
        </div>

      </div>
    </div>
  );
}
