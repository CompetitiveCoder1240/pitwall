"use client";

import React from "react";
import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Bot, User, Flame, ShieldCheck, AlertTriangle } from "lucide-react";
import { Message } from "@/hooks/useSSEChat";

interface ChatMessageProps {
  message: Message;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.role === "user";
  
  // Parse guardrail out of AI messages
  let mainContent = message.content;
  let guardrailLine = null;
  let isGuardrailWarning = false;

  if (!isUser && message.content.includes("[Citation Guardrail]:")) {
    const lines = message.content.split('\n');
    const guardrailIndex = lines.findIndex(line => line.includes("[Citation Guardrail]:"));
    if (guardrailIndex !== -1) {
      guardrailLine = lines[guardrailIndex];
      if (guardrailLine.includes("⚠️") || guardrailLine.includes("Unverified")) {
        isGuardrailWarning = true;
      }
      
      mainContent = lines.slice(0, guardrailIndex).join('\n').trim();
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 12, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      className={`flex gap-3.5 my-4 ${isUser ? "justify-end" : "justify-start"}`}
    >
      {/* AI Avatar */}
      {!isUser && (
        <div className="h-9 w-9 rounded-xl bg-gradient-to-tr from-red-600 to-red-500 flex items-center justify-center shrink-0 shadow-lg shadow-red-600/20 border border-red-400/30">
          <Bot className="h-5 w-5 text-white" />
        </div>
      )}

      {/* Message Content Bubble */}
      <div className={`max-w-3xl space-y-1.5 ${isUser ? "items-end" : "items-start"}`}>
        <div className="flex items-center gap-2 px-1">
          <span className="text-[11px] font-semibold text-zinc-400 font-mono">
            {isUser ? "YOU" : "PITWALL CONSULTANT"}
          </span>
          <span className="text-[10px] text-zinc-600 font-mono">{message.timestamp}</span>
        </div>

        <div
          className={`p-4 rounded-2xl text-xs leading-relaxed ${
            isUser
              ? "bg-gradient-to-r from-red-600 to-red-700 text-white shadow-lg shadow-red-600/20 rounded-tr-none font-medium"
              : "glass-card text-zinc-200 border-white/10 rounded-tl-none"
          }`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="flex flex-col space-y-4">
              <div className="prose prose-invert prose-xs max-w-none prose-p:leading-relaxed prose-pre:bg-zinc-950 prose-pre:border prose-pre:border-zinc-800 prose-table:border-collapse prose-th:border prose-th:border-zinc-700 prose-th:bg-zinc-900 prose-th:p-2 prose-td:border prose-td:border-zinc-800 prose-td:p-2">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {mainContent}
                </ReactMarkdown>

                {message.isStreaming && !guardrailLine && (
                  <span className="inline-block h-4 w-2 bg-red-500 ml-1 animate-pulse" />
                )}
              </div>
              
              {guardrailLine && (
                <div className={`mt-2 flex items-start gap-2 p-3 rounded-lg border ${
                  isGuardrailWarning 
                    ? "bg-amber-950/20 border-amber-500/30 shadow-[0_0_15px_rgba(245,158,11,0.1)] text-amber-200" 
                    : "bg-emerald-950/20 border-emerald-500/30 shadow-[0_0_15px_rgba(16,185,129,0.1)] text-emerald-200"
                }`}>
                  <div className="shrink-0 mt-0.5">
                    {isGuardrailWarning ? (
                      <AlertTriangle className="h-4 w-4 text-amber-500" />
                    ) : (
                      <ShieldCheck className="h-4 w-4 text-emerald-500" />
                    )}
                  </div>
                  <div className="text-[11px] font-mono leading-relaxed">
                    {guardrailLine.replace(/^(🛡️|⚠️)\s*/, '')}
                    {message.isStreaming && (
                      <span className={`inline-block h-3 w-1.5 ml-1 animate-pulse ${
                        isGuardrailWarning ? "bg-amber-500" : "bg-emerald-500"
                      }`} />
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* User Avatar */}
      {isUser && (
        <div className="h-9 w-9 rounded-xl bg-zinc-800 border border-zinc-700 flex items-center justify-center shrink-0">
          <User className="h-4 w-4 text-zinc-300" />
        </div>
      )}
    </motion.div>
  );
};
