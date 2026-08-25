"use client";

import React from "react";
import { Flame, Activity, Cpu, Sparkles, LogOut } from "lucide-react";
import { useRouter } from "next/navigation";

interface HeaderProps {
  backendStatus: "online" | "offline" | "checking";
}

export const Header: React.FC<HeaderProps> = ({ backendStatus }) => {
  const router = useRouter();

  const handleLogout = () => {
    sessionStorage.removeItem("pitwall_jwt");
    router.push("/login");
  };

  return (
    <header className="sticky top-0 z-40 w-full glass-panel border-b border-white/10 px-4 py-3 shadow-2xl">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Left Branding */}
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-red-600 to-red-500 flex items-center justify-center shadow-lg shadow-red-600/30 border border-red-400/30">
            <Flame className="h-6 w-6 text-white animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-['Orbitron'] text-xl font-bold tracking-wider text-white">
                PITWALL <span className="text-red-500 font-normal text-xs px-2 py-0.5 rounded bg-red-950/60 border border-red-500/30">v3 AGENTIC</span>
              </h1>
            </div>
            <p className="text-xs text-zinc-400 flex items-center gap-1.5 mt-0.5">
              <span>FIA 2026 F1 Technical & Sporting Regulations</span>
            </p>
          </div>
        </div>

        {/* Right Telemetry Badges */}
        <div className="flex items-center gap-3">
          {/* LLM Model Badge */}
          <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-900/90 border border-zinc-800 text-xs text-zinc-300">
            <Cpu className="h-3.5 w-3.5 text-red-500" />
            <span className="font-mono text-zinc-200">Gemini 3.5 Flash</span>
          </div>

          {/* Health Status Indicator */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-900/90 border border-zinc-800 text-xs">
            <Activity className="h-3.5 w-3.5 text-zinc-400" />
            <div className="flex items-center gap-1.5">
              {backendStatus === "online" && (
                <>
                  <span className="h-2 w-2 rounded-full bg-emerald-500 pulse-red" />
                  <span className="text-emerald-400 font-medium font-mono">ONLINE</span>
                </>
              )}
              {backendStatus === "offline" && (
                <>
                  <span className="h-2 w-2 rounded-full bg-red-500" />
                  <span className="text-red-400 font-medium font-mono">OFFLINE</span>
                </>
              )}
              {backendStatus === "checking" && (
                <>
                  <span className="h-2 w-2 rounded-full bg-amber-500 animate-ping" />
                  <span className="text-amber-400 font-medium font-mono">CONNECTING</span>
                </>
              )}
            </div>
          </div>

          {/* Logout Button */}
          <button
            onClick={handleLogout}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-900/90 border border-zinc-800 text-xs text-zinc-300 hover:text-white hover:bg-zinc-800 transition-colors"
          >
            <LogOut className="h-3.5 w-3.5" />
            <span className="font-mono hidden sm:inline">LOGOUT</span>
          </button>
        </div>
      </div>
    </header>
  );
};
