import { useState } from "react";
import { HeroSection } from "@/components/HeroSection";
import { GradingForm, type GradingPayload } from "@/components/GradingForm";
import { ResultsSection } from "@/components/ResultsSection";
import { ModelInfoSection } from "@/components/ModelInfoSection";
import { FeaturesSection } from "@/components/FeaturesSection";
import { useToast } from "@/hooks/use-toast";
import {
  newSession, uploadFile, setQuestions, setModelAnswer, setRubricManual,
  startGrading, startGeminiGrading, getResults,
  type GradingResults,
} from "@/lib/api";

const Index = () => {
  const [step, setStep]           = useState<"hero" | "form" | "results">("hero");
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults]     = useState<GradingResults | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [maxMarks, setMaxMarks]   = useState(10);
  const { toast } = useToast();

  const handleSubmit = async ({ questions, files, mode, maxMarks: marks, modelAnswer, rubricCriteria }: GradingPayload) => {
    setIsLoading(true);
    setMaxMarks(marks);
    try {
      const sid = await newSession();
      setSessionId(sid);

      for (const file of files) await uploadFile(sid, file);
      await setQuestions(sid, questions);

      // Standard mode: send user-provided model answer and rubric before grading
      if (mode === "standard") {
        if (modelAnswer) await setModelAnswer(sid, modelAnswer);
        if (rubricCriteria && Object.keys(rubricCriteria).length > 0) {
          await setRubricManual(sid, {
            criteria: Object.entries(rubricCriteria).map(([name, desc]) => ({
              name,
              description: desc,
              max_marks: Math.round(marks / Object.keys(rubricCriteria).length),
              keywords: [],
              performance_levels: { excellent: "Fully meets criterion", good: "Mostly meets criterion", average: "Partially meets criterion", poor: "Does not meet criterion" },
            })),
            total_marks: marks,
            general_instructions: "Evaluate each criterion independently.",
          });
        }
        await startGrading(sid);
      } else {
        await startGeminiGrading(sid);
      }

      const data = await getResults(sid);
      setResults(data);
      setStep("results");
      toast({
        title: "Grading Complete",
        description: `${data.summary.successful_gradings} file(s) graded. Average: ${((data.summary.average_score / 100) * marks).toFixed(1)}/${marks}`,
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      toast({ title: "Grading Failed", description: msg, variant: "destructive" });
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => { setStep("hero"); setResults(null); setSessionId(null); };

  return (
    <div className="min-h-screen bg-background">
      {step === "hero" && (
        <><HeroSection onGetStarted={() => setStep("form")} /><ModelInfoSection /><FeaturesSection /></>
      )}
      {step === "form" && (
        <GradingForm onSubmit={handleSubmit} isLoading={isLoading} />
      )}
      {step === "results" && (
        <ResultsSection results={results} sessionId={sessionId} maxMarks={maxMarks} onReset={handleReset} />
      )}
    </div>
  );
};

export default Index;
