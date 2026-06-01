import { Brain, Zap, Target, Shield, CheckCircle, TrendingUp, Eye, BookOpen } from "lucide-react";
import { Card } from "@/components/ui/card";

export function ModelInfoSection() {
  const features = [
    {
      icon: <Brain className="w-8 h-8 text-primary" />,
      title: "Advanced Neural Networks",
      description: "Powered by state-of-the-art transformer architecture trained on millions of educational assessments"
    },
    {
      icon: <Eye className="w-8 h-8 text-secondary" />,
      title: "Computer Vision Recognition",
      description: "OCR and handwriting recognition with 98.5% accuracy across multiple languages and writing styles"
    },
    {
      icon: <Target className="w-8 h-8 text-accent" />,
      title: "Context-Aware Scoring",
      description: "Understands question context, partial credit scenarios, and subject-specific grading criteria"
    },
    {
      icon: <Shield className="w-8 h-8 text-destructive" />,
      title: "Bias-Free Assessment",
      description: "Trained on diverse datasets with continuous bias detection and fairness optimization"
    }
  ];

  const capabilities = [
    {
      category: "Subject Coverage",
      items: ["Mathematics (K-12 to University)", "Science (Physics, Chemistry, Biology)", "Language Arts & Literature", "History & Social Studies", "Computer Science & Programming"]
    },
    {
      category: "Question Types",
      items: ["Multiple Choice", "Short Answer", "Essay Questions", "Mathematical Proofs", "Diagram Analysis", "Code Review"]
    },
    {
      category: "Input Formats",
      items: ["Handwritten Text", "Typed Responses", "Mathematical Equations", "Diagrams & Charts", "Code Snippets", "Mixed Media"]
    }
  ];

  const metrics = [
    { label: "Accuracy Rate", value: "94.7%", description: "Agreement with human graders" },
    { label: "Processing Speed", value: "<3s", description: "Average grading time" },
    { label: "Languages Supported", value: "25+", description: "Including RTL scripts" },
    { label: "Training Data", value: "50M+", description: "Graded assessments" }
  ];

  return (
    <section className="py-20 px-6 bg-gradient-to-b from-background to-muted/30">
      <div className="container mx-auto max-w-7xl">
        {/* Header */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 bg-primary/10 px-4 py-2 rounded-full mb-6">
            <Brain className="w-5 h-5 text-primary" />
            <span className="text-sm font-semibold text-primary">AI Technology</span>
          </div>
          <h2 className="text-4xl md:text-5xl font-bold mb-6 bg-gradient-to-r from-primary via-secondary to-accent bg-clip-text text-transparent">
            Powered by Advanced AI
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
            Our grading model combines cutting-edge machine learning with educational expertise to deliver 
            precise, fair, and instant assessment capabilities that rival human educators.
          </p>
        </div>

        {/* Core Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-20">
          {features.map((feature, index) => (
            <Card 
              key={index} 
              className="p-6 text-center hover:shadow-medium transition-all duration-300 transform hover:-translate-y-2 bg-card/80 backdrop-blur-sm border-border/50"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <div className="flex justify-center mb-4">
                <div className="w-16 h-16 bg-gradient-hero rounded-full flex items-center justify-center">
                  {feature.icon}
                </div>
              </div>
              <h3 className="text-lg font-semibold mb-3">{feature.title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{feature.description}</p>
            </Card>
          ))}
        </div>

        {/* Capabilities Breakdown */}
        <div className="grid lg:grid-cols-3 gap-8 mb-20">
          {capabilities.map((capability, index) => (
            <Card key={index} className="p-6 shadow-soft bg-card/90 backdrop-blur-sm border-border/50">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <div className="w-2 h-2 bg-gradient-primary rounded-full"></div>
                {capability.category}
              </h3>
              <ul className="space-y-3">
                {capability.items.map((item, itemIndex) => (
                  <li key={itemIndex} className="flex items-start gap-3 text-sm">
                    <CheckCircle className="w-4 h-4 text-secondary mt-0.5 flex-shrink-0" />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </Card>
          ))}
        </div>

        {/* How It Works Process */}
        <Card className="p-8 shadow-medium bg-card/90 backdrop-blur-sm border-border/50">
          <h3 className="text-2xl font-bold text-center mb-8 flex items-center justify-center gap-3">
            <BookOpen className="w-7 h-7 text-primary" />
            How Our AI Grading Works
          </h3>
          
          <div className="grid md:grid-cols-4 gap-6">
            {[
              {
                step: "01",
                title: "Image Processing",
                description: "Advanced OCR extracts text and analyzes handwriting with 98.5% accuracy",
                icon: <Eye className="w-6 h-6" />
              },
              {
                step: "02", 
                title: "Content Analysis",
                description: "NLP models understand context, identify key concepts, and parse mathematical expressions",
                icon: <Brain className="w-6 h-6" />
              },
              {
                step: "03",
                title: "Knowledge Assessment", 
                description: "Compares response against rubrics, identifies correct reasoning, and evaluates completeness",
                icon: <Target className="w-6 h-6" />
              },
              {
                step: "04",
                title: "Score Generation",
                description: "Provides detailed scoring with confidence levels, feedback, and improvement suggestions",
                icon: <TrendingUp className="w-6 h-6" />
              }
            ].map((process, index) => (
              <div key={index} className="text-center relative">
                {/* Connection Line */}
                {index < 3 && (
                  <div className="hidden md:block absolute top-8 left-full w-full h-0.5 bg-gradient-to-r from-primary to-secondary opacity-30 z-0" />
                )}
                
                <div className="relative z-10">
                  <div className="w-16 h-16 bg-gradient-primary rounded-full flex items-center justify-center text-primary-foreground font-bold text-lg mb-4 mx-auto">
                    {process.step}
                  </div>
                  <div className="flex justify-center mb-3">
                    <div className="w-10 h-10 bg-gradient-secondary rounded-full flex items-center justify-center text-secondary-foreground">
                      {process.icon}
                    </div>
                  </div>
                  <h4 className="font-semibold mb-2">{process.title}</h4>
                  <p className="text-sm text-muted-foreground leading-relaxed">{process.description}</p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </section>
  );
}