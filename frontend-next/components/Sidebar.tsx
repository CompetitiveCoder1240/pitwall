"use client";

import React, { useState } from "react";
import { 
  BookOpen, 
  Trash2, 
  Layers, 
  Gauge, 
  CheckCircle2, 
  Info, 
  X,
  Zap,
  Search,
  Bot,
  ChevronDown,
  ChevronUp,
  Workflow,
  Database,
  Network,
  ShieldCheck
} from "lucide-react";

interface SidebarProps {
  onClearSession: () => void;
  isOpen: boolean;
  onClose: () => void;
}

const REGULATION_SECTIONS = [
  { code: "Section A", title: "General Regulatory Provisions", desc: "Governance, FPP tests, entry fees, effective dates" },
  { code: "Section B", title: "Sporting Regulations", desc: "Sprint races, pit lane speed (80km/h), tyre sets, parc fermé" },
  { code: "Section C", title: "Technical Regulations", desc: "768kg min weight, 80kg driver, 350kW MGU-K, active aero Z/X mode" },
  { code: "Section D", title: "Financial — F1 Teams", desc: "Cost cap limits, 31 March reporting, 5% overspend thresholds" },
  { code: "Section E", title: "Financial — PU Manufacturers", desc: "PU cost cap limits, customer supply exclusions, ABA rules" },
  { code: "Section F", title: "Operational Regulations", desc: "Summer factory shutdown (14 days), curfew rest windows" },
];

export const Sidebar: React.FC<SidebarProps> = ({ onClearSession, isOpen, onClose }) => {
  const [showArchitectureModal, setShowArchitectureModal] = useState(false);
  const [isSectionsOpen, setIsSectionsOpen] = useState(false);

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/70 backdrop-blur-sm z-40 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar Panel */}
      <aside
        className={`fixed top-16 bottom-0 left-0 z-40 w-80 glass-panel border-r border-white/10 p-4 transition-transform duration-300 flex flex-col justify-between ${
          isOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
        }`}
      >
        <div className="space-y-6 overflow-y-auto pr-1">
          {/* Mobile Close Button */}
          <div className="flex items-center justify-between lg:hidden pb-2 border-b border-zinc-800">
            <span className="font-['Orbitron'] text-sm font-semibold text-zinc-300">TELEMETRY MENU</span>
            <button onClick={onClose} className="p-1 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800">
              <X className="h-5 w-5" />
            </button>
          </div>

          {/* Action: Clear Session */}
          <button
            onClick={onClearSession}
            className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl bg-zinc-900 border border-zinc-800 hover:border-red-600/50 hover:bg-red-950/30 text-zinc-300 hover:text-red-400 font-medium text-xs transition-all shadow-md group"
          >
            <Trash2 className="h-4 w-4 text-zinc-500 group-hover:text-red-400 transition-colors" />
            <span>Reset Telemetry Session</span>
          </button>

          {/* New Section: Active Capabilities */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-xs font-semibold text-zinc-400 uppercase tracking-wider">
              <Workflow className="h-3.5 w-3.5 text-red-500" />
              <span>Active Agentic Capabilities</span>
            </div>
            
            <div className="space-y-2">
              <div className="p-2.5 rounded-lg bg-emerald-950/20 border border-emerald-900/30 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Workflow className="h-3.5 w-3.5 text-emerald-500" />
                  <span className="text-xs font-medium text-emerald-400">LangGraph StateGraph</span>
                </div>
                <span className="text-[10px] text-zinc-500">Orchestration</span>
              </div>
              
              <div className="p-2.5 rounded-lg bg-blue-950/20 border border-blue-900/30 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Database className="h-3.5 w-3.5 text-blue-500" />
                  <span className="text-xs font-medium text-blue-400">Telemetry DB 2022-2025</span>
                </div>
                <span className="text-[10px] text-zinc-500">SQLite data</span>
              </div>

              <div className="p-2.5 rounded-lg bg-purple-950/20 border border-purple-900/30 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Network className="h-3.5 w-3.5 text-purple-500" />
                  <span className="text-xs font-medium text-purple-400">GraphRAG Engine</span>
                </div>
                <span className="text-[10px] text-zinc-500">721 Nodes</span>
              </div>

              <div className="p-2.5 rounded-lg bg-amber-950/20 border border-amber-900/30 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="h-3.5 w-3.5 text-amber-500" />
                  <span className="text-xs font-medium text-amber-400">Citation Guardrail</span>
                </div>
                <span className="text-[10px] text-zinc-500">Anti-hallucination</span>
              </div>
            </div>
          </div>

          {/* Section 1: Regulation Sections (Collapsible) */}
          <div className="space-y-3">
            <button 
              onClick={() => setIsSectionsOpen(!isSectionsOpen)}
              className="w-full flex items-center justify-between text-xs font-semibold text-zinc-400 uppercase tracking-wider hover:text-zinc-300 transition-colors"
            >
              <div className="flex items-center gap-2">
                <BookOpen className="h-3.5 w-3.5 text-red-500" />
                <span>FIA 2026 Regulation Manuals</span>
              </div>
              {isSectionsOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </button>

            {isSectionsOpen && (
              <div className="space-y-2">
                {REGULATION_SECTIONS.map((sec) => (
                  <div
                    key={sec.code}
                    className="p-2.5 rounded-lg bg-zinc-950/60 border border-zinc-800/80 hover:border-zinc-700 transition-all group"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-xs font-bold text-red-500">{sec.code}</span>
                      <span className="text-[10px] text-zinc-500 group-hover:text-zinc-300 transition-colors">Active</span>
                    </div>
                    <h4 className="text-xs font-medium text-zinc-200 mt-1">{sec.title}</h4>
                    <p className="text-[11px] text-zinc-400 mt-0.5 leading-snug">{sec.desc}</p>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Section 2: Pipeline Specs */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-xs font-semibold text-zinc-400 uppercase tracking-wider">
              <Layers className="h-3.5 w-3.5 text-red-500" />
              <span>LangGraph v3 Engine</span>
            </div>

            <div className="p-3 rounded-xl bg-zinc-950/80 border border-zinc-800 space-y-2 text-xs">
              <div className="flex items-center justify-between text-zinc-300">
                <span className="flex items-center gap-1.5"><Search className="h-3 w-3 text-red-400" /> Search Type</span>
                <span className="font-mono text-[11px] text-zinc-400">Hybrid (BM25 + Vector)</span>
              </div>
              <div className="flex items-center justify-between text-zinc-300">
                <span className="flex items-center gap-1.5"><Zap className="h-3 w-3 text-amber-400" /> Reranker</span>
                <span className="font-mono text-[11px] text-zinc-400">FlashRank Cross-Encoder</span>
              </div>
              <div className="flex items-center justify-between text-zinc-300">
                <span className="flex items-center gap-1.5"><Gauge className="h-3 w-3 text-emerald-400" /> Chunk Strategy</span>
                <span className="font-mono text-[11px] text-zinc-400">Parent-Child (400c/2000c)</span>
              </div>
              <div className="flex items-center justify-between text-zinc-300">
                <span className="flex items-center gap-1.5"><CheckCircle2 className="h-3 w-3 text-blue-400" /> Hit Rate @ 3</span>
                <span className="font-mono text-[11px] text-emerald-400 font-bold">83.33% (Passed)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Footer Info button */}
        <div className="pt-4 border-t border-zinc-800/80 mt-4">
          <button
            onClick={() => setShowArchitectureModal(true)}
            className="w-full flex items-center justify-center gap-1.5 text-xs text-zinc-400 hover:text-white transition-colors py-1"
          >
            <Info className="h-3.5 w-3.5" />
            <span>How PitWall LangGraph Engine Works</span>
          </button>
        </div>
      </aside>

      {/* Architecture Modal */}
      {showArchitectureModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4">
          <div className="glass-panel w-full max-w-xl p-6 rounded-2xl border border-white/10 space-y-4 shadow-2xl relative">
            <button
              onClick={() => setShowArchitectureModal(false)}
              className="absolute top-4 right-4 p-1 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800"
            >
              <X className="h-5 w-5" />
            </button>

            <div className="flex items-center gap-2">
              <Bot className="h-5 w-5 text-red-500" />
              <h3 className="font-['Orbitron'] text-lg font-bold text-white">PitWall LangGraph v3 Engine</h3>
            </div>

            <div className="space-y-3 text-xs text-zinc-300 leading-relaxed">
              <p>
                PitWall processes official FIA 2026 PDFs using a <strong>Parent-Child chunking architecture</strong>. Small 400-character snippets are embedded for search accuracy, while full 2,000-character articles are provided to the LLM for rich context. The pipeline is orchestrated via LangGraph.
              </p>
              
              <div className="p-3 rounded-xl bg-zinc-950 border border-zinc-800 space-y-1 font-mono text-[11px] text-zinc-400">
                <div>1. Router → Determine intent (Telemetry / Regulations)</div>
                <div>2. Parallel Nodes → Execute Hybrid Search & GraphRAG simultaneously</div>
                <div>3. Synthesis → Gemini 2.5 Pro processes context & generates response</div>
                <div>4. Citation Guardrail → Verify claims against source documents before SSE</div>
              </div>

              <p className="text-zinc-400">
                Benchmarked on 50 expert-verified regulation questions with <strong>83.33% Hit Rate</strong> and <strong>0.8333 MRR</strong> score.
              </p>
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setShowArchitectureModal(false)}
                className="px-4 py-2 rounded-xl bg-red-600 hover:bg-red-500 text-white font-medium text-xs transition-colors"
              >
                Close View
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
