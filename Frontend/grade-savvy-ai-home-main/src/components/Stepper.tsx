import { Check } from "lucide-react";

export interface Step {
  id: number;
  label: string;
  description: string;
}

interface StepperProps {
  steps: Step[];
  current: number;
}

export function Stepper({ steps, current }: StepperProps) {
  return (
    <div className="w-full px-6 py-8">
      <div className="flex items-start justify-between relative">
        {/* connecting line */}
        <div className="absolute top-5 left-0 right-0 h-0.5 bg-border mx-10 z-0" />
        <div
          className="absolute top-5 left-0 h-0.5 bg-primary z-0 transition-all duration-500 mx-10"
          style={{ width: `calc(${((current - 1) / (steps.length - 1)) * 100}% - 0px)` }}
        />

        {steps.map((step) => {
          const done    = step.id < current;
          const active  = step.id === current;
          return (
            <div key={step.id} className="flex flex-col items-center z-10 flex-1">
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all duration-300 font-semibold text-sm
                  ${done   ? "bg-primary border-primary text-primary-foreground"
                  : active ? "bg-background border-primary text-primary shadow-[0_0_0_4px_hsl(var(--primary)/0.15)]"
                           : "bg-background border-border text-muted-foreground"}`}
              >
                {done ? <Check className="w-5 h-5" /> : step.id}
              </div>
              <p className={`mt-2 text-xs font-semibold text-center leading-tight
                ${active ? "text-primary" : done ? "text-foreground" : "text-muted-foreground"}`}>
                {step.label}
              </p>
              <p className="text-[10px] text-muted-foreground text-center hidden sm:block mt-0.5">
                {step.description}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
