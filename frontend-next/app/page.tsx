"use client";

import React, { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Send, Menu, Sparkles, ShieldCheck, Gauge, Zap, BookOpen, Activity, Flame } from "lucide-react";
import { Header } from "@/components/Header";
import { Sidebar } from "@/components/Sidebar";
import { ChatMessage } from "@/components/ChatMessage";
import { SuggestionCard } from "@/components/SuggestionCard";
import { useSSEChat } from "@/hooks/useSSEChat";

const STARTER_SUGGESTIONS = [
  {
    category: 'RACE STRATEGY',
    title: 'Safety Car pit strategy at Monza on lap 15',
    query: 'What strategy should I use at Monza if I am on lap 15 out of 53 under a safety car? Should I pit now?',
    icon: <Gauge className="h-4 w-4" />,
  },
  {
    category: 'REGULATIONS',
    title: 'Parc Fermé front wing replacement penalty',
    query: 'What is the penalty for changing a front wing under parc ferme conditions?',
    icon: <ShieldCheck className="h-4 w-4" />,
  },
  {
    category: 'POWERTRAIN',
    title: '2026 MGU-K power and active aero rules',
    query: 'What is the maximum electrical power output for the MGU-K in the 2026 regulations and how does active aero work?',
    icon: <Zap className="h-4 w-4" />,
  },
  {
    category: 'TELEMETRY',
    title: 'Tyre degradation comparison at Silverstone',
    query: 'Compare Soft vs Medium tyre degradation rates at Silverstone from 2022 to 2025 telemetry data.',
    icon: <Activity className="h-4 w-4" />,
  }
];

export default function Home() {
  const router = useRouter();
  const { messages, isLoading, backendStatus, sendMessage, clearSession } = useSSEChat();
  const [inputValue, setInputValue] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Authentication check
  useEffect(() => {
    const token = sessionStorage.getItem("pitwall_jwt");
    if (!token) {
      router.push("/login");
    }
  }, [router]);

  // Auto-scroll to bottom of chat feed
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!inputValue.trim() || isLoading) return;
    sendMessage(inputValue);
    setInputValue("");
  };

  const handleSuggestionSelect = (query: string) => {
    sendMessage(query);
  };

  return (
    <div className="flex flex-col min-h-screen bg-[#070709] text-zinc-100 selection:bg-red-600/30 selection:text-white">
      {/* Top Telemetry Header */}
      <Header backendStatus={backendStatus} />

      {/* Main Layout Area */}
      <div className="flex flex-1 pt-2">
        {/* Sidebar Navigation */}
        <Sidebar
          isOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          onClearSession={clearSession}
        />

        {/* Content Viewport */}
        <main className="flex-1 flex flex-col justify-between max-w-4xl mx-auto w-full px-4 pb-6 lg:pl-84 transition-all">
          
          {/* Mobile Menu Bar Toggle */}
          <div className="lg:hidden flex items-center justify-between py-2 px-1 mb-2 border-b border-zinc-800/80">
            <button
              onClick={() => setSidebarOpen(true)}
              className="flex items-center gap-2 py-1.5 px-3 rounded-lg bg-zinc-900 border border-zinc-800 text-xs font-mono text-zinc-300 hover:text-white"
            >
              <Menu className="h-4 w-4 text-red-500" />
              <span>TELEMETRY MENU</span>
            </button>
            <span className="text-[11px] font-mono text-zinc-500">FIA 2026 REGS</span>
          </div>

          {/* Empty State / Welcome Screen */}
          {messages.length === 0 ? (
            <div className="flex-1 flex flex-col justify-center items-center py-10 space-y-8">
              
              {/* Hero Emblem */}
              <div className="text-center space-y-3 max-w-lg">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-red-950/60 border border-red-500/30 text-red-400 text-xs font-mono mb-2 shadow-lg shadow-red-950/40">
                  <Sparkles className="h-3.5 w-3.5" />
                  <span>RACE STRATEGY & REGULATION CONSULTANT</span>
                </div>
                <h2 className="font-['Orbitron'] text-3xl font-bold tracking-tight text-white sm:text-4xl">
                  READY ON THE <span className="f1-text-gradient font-extrabold">PITWALL</span>
                </h2>
                <p className="text-xs text-zinc-400 leading-relaxed">
                  AI-powered F1 race strategy, regulation compliance, and telemetry analysis.
                </p>
              </div>

              {/* Starter Suggestions Grid */}
              <div className="w-full max-w-2xl space-y-3">
                <div className="flex items-center justify-between px-1">
                  <span className="text-xs font-mono text-zinc-400 flex items-center gap-1.5">
                    <Activity className="h-3.5 w-3.5 text-red-500" /> RECOMMENDED TELEMETRY PROMPTS
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {STARTER_SUGGESTIONS.map((s, idx) => (
                    <SuggestionCard
                      key={idx}
                      category={s.category}
                      title={s.title}
                      query={s.query}
                      icon={s.icon}
                      onSelect={handleSuggestionSelect}
                    />
                  ))}
                </div>
              </div>
            </div>
          ) : (
            /* Chat Feed Messages Area */
            <div className="flex-1 overflow-y-auto py-4 space-y-2">
              {messages.map((msg) => (
                <ChatMessage key={msg.id} message={msg} />
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}

          {/* Bottom Fixed Streaming Input Bar */}
          <div className="mt-4 sticky bottom-4 z-30">
            <form
              onSubmit={handleSubmit}
              className="glass-panel p-2 rounded-2xl border border-white/10 shadow-2xl focus-within:border-red-500/50 focus-within:ring-2 focus-within:ring-red-500/20 transition-all duration-300"
            >
              <div className="flex items-center gap-2 px-2">
                <div className="p-2 rounded-xl bg-zinc-900 text-red-500 hidden sm:block">
                  <Flame className="h-4 w-4" />
                </div>

                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="Ask a race strategy, regulation, or telemetry question..."
                  className="flex-1 bg-transparent border-none outline-none text-xs text-white placeholder:text-zinc-500 py-2 font-sans"
                  disabled={isLoading}
                />

                <button
                  type="submit"
                  disabled={!inputValue.trim() || isLoading}
                  className="p-2.5 rounded-xl bg-gradient-to-r from-red-600 to-red-500 hover:from-red-500 hover:to-red-400 text-white shadow-lg shadow-red-600/30 disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200"
                >
                  <Send className="h-4 w-4" />
                </button>
              </div>
            </form>

            <div className="flex items-center justify-between px-3 mt-2 text-[10px] font-mono text-zinc-400">
              <span>AI Consultant</span>
              <span>Active Guardrail</span>
            </div>
          </div>

        </main>
      </div>
    </div>
  );
}
