import { Bot, GraduationCap, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";

interface HeroSectionProps {
  onGetStarted: () => void;
}

export function HeroSection({ onGetStarted }: HeroSectionProps) {
  return (
    <section className="relative min-h-[80vh] flex items-center justify-center bg-gradient-hero overflow-hidden">
      {/* Floating Elements */}
      <div className="absolute top-20 left-10 animate-float">
        <div className="w-16 h-16 bg-gradient-primary rounded-full opacity-60" />
      </div>
      <div className="absolute top-32 right-20 animate-float" style={{ animationDelay: '1s' }}>
        <div className="w-12 h-12 bg-gradient-secondary rounded-full opacity-50" />
      </div>
      <div className="absolute bottom-20 left-1/4 animate-float" style={{ animationDelay: '2s' }}>
        <div className="w-8 h-8 bg-gradient-accent rounded-full opacity-40" />
      </div>

      <div className="container mx-auto px-6 text-center z-10">
        {/* Main Icon */}
        <div className="mb-8 flex justify-center">
          <div className="relative">
            <div className="w-24 h-24 bg-gradient-primary rounded-full flex items-center justify-center shadow-glow animate-pulse-glow">
              <Bot className="w-12 h-12 text-primary-foreground" />
            </div>
            <div className="absolute -top-2 -right-2 w-8 h-8 bg-gradient-accent rounded-full flex items-center justify-center">
              <GraduationCap className="w-4 h-4 text-accent-foreground" />
            </div>
          </div>
        </div>

        {/* Headline */}
        <h1 className="text-5xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent leading-tight">
          GradeSavvy AI
        </h1>
        
        <p className="text-xl md:text-2xl text-muted-foreground mb-6 max-w-3xl mx-auto leading-relaxed">
          Revolutionary AI-powered grading assistant that evaluates student work with precision and provides instant feedback. 
          Upload questions and student responses to get accurate scores out of 10.
        </p>


        {/* Features */}
        <div className="flex flex-wrap justify-center gap-6 mb-10 max-w-4xl mx-auto">
          <div className="flex items-center gap-2 bg-card/50 px-4 py-2 rounded-full backdrop-blur-sm border border-border/50">
            <Zap className="w-5 h-5 text-primary" />
            <span className="text-sm font-medium">Instant Scoring</span>
          </div>
          <div className="flex items-center gap-2 bg-card/50 px-4 py-2 rounded-full backdrop-blur-sm border border-border/50">
            <Bot className="w-5 h-5 text-secondary" />
            <span className="text-sm font-medium">AI-Powered Analysis</span>
          </div>
          <div className="flex items-center gap-2 bg-card/50 px-4 py-2 rounded-full backdrop-blur-sm border border-border/50">
            <GraduationCap className="w-5 h-5 text-accent" />
            <span className="text-sm font-medium">Educational Insights</span>
          </div>
        </div>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Button 
            variant="hero" 
            size="xl" 
            onClick={onGetStarted}
            className="min-w-48"
          >
            Start Grading Now
          </Button>
          <Button 
            variant="outline" 
            size="xl"
            className="min-w-48 border-border/30 bg-card/30 backdrop-blur-sm hover:bg-card/50"
          >
            Learn More
          </Button>
        </div>
      </div>
    </section>
  );
}