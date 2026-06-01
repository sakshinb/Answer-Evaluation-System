import { Sparkles, Clock, Users, Cpu } from "lucide-react";
import { Card } from "@/components/ui/card";

export function FeaturesSection() {
  const features = [
    {
      icon: <Sparkles className="w-12 h-12 text-primary" />,
      title: "Instant Feedback",
      description: "Get comprehensive grading results in under 3 seconds with detailed explanations and improvement suggestions.",
      color: "from-primary to-primary-glow"
    },
    {
      icon: <Clock className="w-12 h-12 text-secondary" />,
      title: "24/7 Availability", 
      description: "Grade assignments any time, anywhere. No waiting for teacher availability or office hours.",
      color: "from-secondary to-secondary-glow"
    },
    {
      icon: <Users className="w-12 h-12 text-accent" />,
      title: "Consistent Grading",
      description: "Eliminate grading bias and ensure fair, consistent evaluation across all students and assignments.",
      color: "from-accent to-accent-glow"
    },
    {
      icon: <Cpu className="w-12 h-12 text-secondary" />,
      title: "Scalable Infrastructure",
      description: "Handle thousands of submissions simultaneously with enterprise-grade reliability.",
      color: "from-secondary to-green-400"
    }
  ];

  return (
    <section className="py-20 px-6 bg-background">
      <div className="container mx-auto max-w-7xl">
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 bg-accent/10 px-4 py-2 rounded-full mb-6">
            <Sparkles className="w-5 h-5 text-accent" />
            <span className="text-sm font-semibold text-accent">Key Benefits</span>
          </div>
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            Why Choose{" "}
            <span className="bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
              GradeSavvy AI
            </span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Experience the future of educational assessment with our advanced AI technology 
            that delivers precision, speed, and fairness in every evaluation.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-2 gap-8 max-w-4xl mx-auto">
          {features.map((feature, index) => (
            <Card 
              key={index}
              className="group p-8 hover:shadow-glow transition-all duration-500 transform hover:-translate-y-3 bg-card/80 backdrop-blur-sm border-border/50 overflow-hidden relative"
              style={{ animationDelay: `${index * 150}ms` }}
            >
              {/* Background Gradient Effect */}
              <div className={`absolute inset-0 bg-gradient-to-br ${feature.color} opacity-0 group-hover:opacity-5 transition-opacity duration-500`} />
              
              <div className="relative z-10">
                <div className="flex justify-center mb-6">
                  <div className={`w-20 h-20 bg-gradient-to-br ${feature.color} rounded-2xl flex items-center justify-center transform group-hover:scale-110 transition-transform duration-300`}>
                    {feature.icon}
                  </div>
                </div>
                
                <h3 className="text-xl font-bold mb-4 text-center group-hover:text-primary transition-colors duration-300">
                  {feature.title}
                </h3>
                
                <p className="text-muted-foreground leading-relaxed text-center">
                  {feature.description}
                </p>
              </div>

              {/* Hover Animation Border */}
              <div className="absolute inset-0 rounded-lg border-2 border-transparent group-hover:border-primary/20 transition-colors duration-300" />
            </Card>
          ))}
        </div>

        {/* Bottom CTA */}
        <div className="text-center mt-16">
          <div className="inline-flex items-center gap-3 bg-gradient-hero px-8 py-4 rounded-2xl border border-border/30">
            <div className="flex -space-x-2">
              <div className="w-8 h-8 bg-gradient-primary rounded-full border-2 border-background" />
              <div className="w-8 h-8 bg-gradient-secondary rounded-full border-2 border-background" />
              <div className="w-8 h-8 bg-gradient-accent rounded-full border-2 border-background" />
            </div>
            <div className="text-left">
              <p className="text-sm font-semibold">Trusted by 10,000+ educators</p>
              <p className="text-xs text-muted-foreground">Processing 1M+ assessments monthly</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}