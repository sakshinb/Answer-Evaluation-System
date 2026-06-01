import { useState } from "react";
import { Bot, GraduationCap, Shield, BookOpen, User, Eye, EyeOff, LogIn } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { useAuth, type Role } from "@/context/AuthContext";
import { useToast } from "@/hooks/use-toast";

const ROLES: { key: Role; label: string; icon: React.ReactNode; color: string; desc: string }[] = [
  {
    key: "admin",
    label: "Admin",
    icon: <Shield className="w-6 h-6" />,
    color: "from-destructive to-red-400",
    desc: "Full system access",
  },
  {
    key: "teacher",
    label: "Teacher",
    icon: <BookOpen className="w-6 h-6" />,
    color: "from-primary to-blue-400",
    desc: "Grade & manage assignments",
  },
  {
    key: "student",
    label: "Student",
    icon: <User className="w-6 h-6" />,
    color: "from-secondary to-green-400",
    desc: "View your results",
  },
];

// Demo credentials per role
const DEMO_USERS: Record<Role, { email: string; password: string; name: string }> = {
  admin:   { email: "admin@gradesavvy.ai",   password: "admin123",   name: "Admin" },
  teacher: { email: "teacher@gradesavvy.ai", password: "teacher123", name: "Teacher" },
  student: { email: "student@gradesavvy.ai", password: "student123", name: "Student" },
};

export function LoginPage() {
  const [role, setRole]       = useState<Role>("teacher");
  const [email, setEmail]     = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw]   = useState(false);
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const { toast } = useToast();

  const fillDemo = () => {
    setEmail(DEMO_USERS[role].email);
    setPassword(DEMO_USERS[role].password);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim() || !password.trim()) {
      toast({ title: "Required", description: "Enter email and password", variant: "destructive" });
      return;
    }
    setLoading(true);
    try {
      const res = await fetch("/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password, role }),
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.message);
      login({ name: data.name, email: data.email, role: data.role });
      toast({ title: `Welcome, ${data.name}!`, description: `Logged in as ${data.role}` });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Login failed";
      toast({ title: "Login Failed", description: msg, variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  const selected = ROLES.find(r => r.key === role)!;

  return (
    <div className="min-h-screen bg-gradient-hero flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-md space-y-6">

        {/* Logo */}
        <div className="text-center">
          <div className="flex justify-center mb-4">
            <div className="relative">
              <div className="w-16 h-16 bg-gradient-primary rounded-full flex items-center justify-center shadow-glow">
                <Bot className="w-8 h-8 text-primary-foreground" />
              </div>
              <div className="absolute -top-1 -right-1 w-6 h-6 bg-gradient-accent rounded-full flex items-center justify-center">
                <GraduationCap className="w-3 h-3 text-accent-foreground" />
              </div>
            </div>
          </div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
            GradeSavvy AI
          </h1>
          <p className="text-muted-foreground text-sm mt-1">Sign in to your account</p>
        </div>

        {/* Role selector */}
        <div className="grid grid-cols-3 gap-3">
          {ROLES.map(r => (
            <button
              key={r.key}
              type="button"
              onClick={() => { setRole(r.key); setEmail(""); setPassword(""); }}
              className={`flex flex-col items-center gap-2 p-4 rounded-xl border-2 transition-all duration-200
                ${role === r.key
                  ? "border-primary bg-primary/8 shadow-soft"
                  : "border-border hover:border-primary/40 bg-card/60"}`}
            >
              <div className={`w-10 h-10 rounded-full bg-gradient-to-br ${r.color} flex items-center justify-center text-white`}>
                {r.icon}
              </div>
              <span className="text-xs font-semibold">{r.label}</span>
              <span className="text-[10px] text-muted-foreground text-center leading-tight">{r.desc}</span>
            </button>
          ))}
        </div>

        {/* Login form */}
        <Card className="p-6 bg-card/90 backdrop-blur-sm border-border/50 shadow-medium">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="flex items-center gap-2 mb-2">
              <div className={`w-7 h-7 rounded-full bg-gradient-to-br ${selected.color} flex items-center justify-center text-white`}>
                {selected.icon}
              </div>
              <span className="font-semibold text-sm">{selected.label} Login</span>
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="email" className="text-sm">Email</Label>
              <Input
                id="email"
                type="email"
                placeholder={`${role}@gradesavvy.ai`}
                value={email}
                onChange={e => setEmail(e.target.value)}
                disabled={loading}
                autoComplete="email"
              />
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="password" className="text-sm">Password</Label>
              <div className="relative">
                <Input
                  id="password"
                  type={showPw ? "text" : "password"}
                  placeholder="••••••••"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  disabled={loading}
                  autoComplete="current-password"
                  className="pr-10"
                />
                <button
                  type="button"
                  onClick={() => setShowPw(v => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                >
                  {showPw ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <Button type="submit" variant="hero" className="w-full" disabled={loading}>
              {loading
                ? <><div className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin mr-2" />Signing in...</>
                : <><LogIn className="w-4 h-4 mr-2" />Sign In</>}
            </Button>
          </form>

          <div className="mt-4 pt-4 border-t border-border/30">
            <p className="text-xs text-muted-foreground text-center mb-2">Demo credentials</p>
            <button
              type="button"
              onClick={fillDemo}
              className="w-full text-xs text-primary hover:underline text-center"
            >
              Fill demo {role} credentials
            </button>
            <div className="mt-2 bg-muted/30 rounded-lg p-2 text-[11px] text-muted-foreground space-y-0.5">
              <p>Email: <span className="font-mono">{DEMO_USERS[role].email}</span></p>
              <p>Password: <span className="font-mono">{DEMO_USERS[role].password}</span></p>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
