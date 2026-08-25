"use client";

import React from "react";
import { ArrowUpRight } from "lucide-react";

interface SuggestionCardProps {
  category: string;
  title: string;
  query: string;
  icon: React.ReactNode;
  onSelect: (query: string) => void;
}

export const SuggestionCard: React.FC<SuggestionCardProps> = ({
  category,
  title,
  query,
  icon,
  onSelect,
}) => {
  return (
    <button
      onClick={() => onSelect(query)}
      className="group text-left glass-card p-4 rounded-2xl flex flex-col justify-between h-32 relative overflow-hidden transition-all duration-300 hover:border-red-500/50"
    >
      {/* Background Subtle Gradient */}
      <div className="absolute top-0 right-0 w-24 h-24 bg-red-600/5 rounded-full blur-2xl group-hover:bg-red-600/15 transition-all duration-300" />

      <div className="flex items-center justify-between w-full">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-xl bg-zinc-900/90 border border-zinc-800 text-red-500 group-hover:text-red-400 group-hover:border-red-500/30 transition-colors">
            {icon}
          </div>
          <span className="font-mono text-[10px] font-bold text-zinc-400 uppercase tracking-wider">
            {category}
          </span>
        </div>
        <ArrowUpRight className="h-4 w-4 text-zinc-600 group-hover:text-red-400 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all" />
      </div>

      <div>
        <h4 className="text-xs font-semibold text-zinc-200 group-hover:text-white transition-colors line-clamp-2 leading-snug">
          {title}
        </h4>
      </div>
    </button>
  );
};
