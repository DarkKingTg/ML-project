"use client";

import React, { useState, useEffect } from "react";
import {
  ShieldAlert,
  ShieldCheck,
  Zap,
  Cpu,
  Sparkles,
  ArrowRight,
  Layers,
  Code2,
  Copy,
  Check,
  Terminal,
} from "lucide-react";
import { PredictionResult, AttackPreset } from "@/types/ml";
import { ATTACK_PRESETS, runClientInference } from "@/lib/ml-engine";
import { classifyPrompt } from "@/lib/api-client";

interface InferencePlaygroundProps {
  onSelectFeatureTab?: () => void;
  onSelectArchitectureTab?: () => void;
  currentResult: PredictionResult | null;
  setCurrentResult: (res: PredictionResult) => void;
}

export function InferencePlayground({
  onSelectFeatureTab,
  onSelectArchitectureTab,
  currentResult,
  setCurrentResult,
}: InferencePlaygroundProps) {
  const [promptText, setPromptText] = useState<string>(ATTACK_PRESETS[0].prompt);
  const [selectedPresetId, setSelectedPresetId] = useState<string>(ATTACK_PRESETS[0].id);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [showTokenOverlay, setShowTokenOverlay] = useState<boolean>(true);
  const [copied, setCopied] = useState<boolean>(false);

  const handleAnalyze = async (textToAnalyze: string = promptText) => {
    if (!textToAnalyze.trim()) return;
    setIsAnalyzing(true);
    try {
      const res = await classifyPrompt(textToAnalyze);
      setCurrentResult(res);
    } catch {
      const fallback = runClientInference(textToAnalyze);
      setCurrentResult(fallback);
    } finally {
      setIsAnalyzing(false);
    }
  };

  useEffect(() => {
    handleAnalyze(ATTACK_PRESETS[0].prompt);
  }, []);

  const handleSelectPreset = (preset: AttackPreset) => {
    setSelectedPresetId(preset.id);
    setPromptText(preset.prompt);
    handleAnalyze(preset.prompt);
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(promptText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isMalicious = currentResult?.prediction === "Malicious";
  const safeProb = Math.round((currentResult?.probabilities.Safe ?? 0.5) * 100);
  const maliciousProb = Math.round((currentResult?.probabilities.Malicious ?? 0.5) * 100);

  // SVG Circular Gauge helper
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const malStrokeDash = circumference - (maliciousProb / 100) * circumference;
  const safeStrokeDash = circumference - (safeProb / 100) * circumference;

  return (
    <div className="space-y-6">
      {/* Top Banner / Attack Preset Carousel */}
      <div className="glass-panel rounded-2xl p-4 sm:p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3">
          <div>
            <h2 className="text-sm font-semibold tracking-tight uppercase text-[var(--text-secondary)]">
              Interactive Attack Library
            </h2>
            <p className="text-xs text-[var(--text-muted)]">
              Select pre-configured jailbreak payloads, system prompt injections, or benign queries.
            </p>
          </div>
          <span className="text-[11px] font-mono text-[var(--accent-cyan)] bg-blue-500/10 px-2.5 py-1 rounded-full border border-blue-500/20 self-start sm:self-auto">
            {ATTACK_PRESETS.length} Test Vectors Ready
          </span>
        </div>

        {/* Preset Chips */}
        <div className="flex flex-wrap gap-2">
          {ATTACK_PRESETS.map((preset) => {
            const isSelected = selectedPresetId === preset.id;
            const isAttack = preset.expectedClass === "Malicious";
            return (
              <button
                key={preset.id}
                onClick={() => handleSelectPreset(preset)}
                className={`px-3 py-1.5 rounded-xl text-xs font-medium transition-all duration-200 flex items-center gap-1.5 border ${
                  isSelected
                    ? isAttack
                      ? "bg-red-500/20 text-red-400 border-red-500/40 shadow-sm"
                      : "bg-emerald-500/20 text-emerald-400 border-emerald-500/40 shadow-sm"
                    : "bg-black/5 dark:bg-white/5 border-[var(--border-subtle)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--border-active)]"
                }`}
              >
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    isAttack ? "bg-red-500" : "bg-emerald-500"
                  }`}
                />
                <span>{preset.title}</span>
                <span className="text-[10px] opacity-70 font-mono">
                  [{preset.category}]
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Grid: Left Editor & Right Neural Assessment */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Prompt Input & Lexical Trigger Overlay (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="glass-panel rounded-2xl p-5 relative overflow-hidden flex flex-col h-full">
            {/* Header controls */}
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Terminal className="w-4 h-4 text-blue-400" />
                <span className="text-sm font-semibold tracking-tight">
                  Prompt Input & Live Tokenizer
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setShowTokenOverlay(!showTokenOverlay)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors border ${
                    showTokenOverlay
                      ? "bg-blue-500/15 text-blue-400 border-blue-500/30"
                      : "bg-black/5 dark:bg-white/5 text-[var(--text-muted)] border-transparent"
                  }`}
                >
                  {showTokenOverlay ? "Hide Triggers" : "Highlight Triggers"}
                </button>
                <button
                  onClick={handleCopy}
                  title="Copy Prompt"
                  className="p-1.5 rounded-lg text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors hover:bg-black/5 dark:hover:bg-white/5"
                >
                  {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {/* In-text Trigger Highlighting View */}
            {showTokenOverlay && currentResult?.telemetry.token_highlights && (
              <div className="p-3 mb-3 rounded-xl bg-black/10 dark:bg-black/40 border border-[var(--border-subtle)] text-xs leading-relaxed max-h-36 overflow-y-auto">
                <div className="text-[10px] uppercase font-bold text-[var(--text-muted)] tracking-wider mb-1.5 flex items-center justify-between">
                  <span>Inline Lexicon Scan:</span>
                  <div className="flex gap-2">
                    <span className="text-red-400 font-normal">■ Override</span>
                    <span className="text-amber-400 font-normal">■ Jailbreak</span>
                    <span className="text-purple-400 font-normal">■ Roleplay</span>
                    <span className="text-cyan-400 font-normal">■ Security</span>
                  </div>
                </div>
                <p className="font-mono whitespace-pre-wrap">
                  {currentResult.telemetry.token_highlights.map((tok, i) => {
                    if (tok.category === "override") {
                      return (
                        <mark key={i} className="bg-red-500/25 text-red-300 font-bold px-1 rounded mx-0.5 border border-red-500/40">
                          {tok.text}
                        </mark>
                      );
                    }
                    if (tok.category === "jailbreak") {
                      return (
                        <mark key={i} className="bg-amber-500/25 text-amber-300 font-bold px-1 rounded mx-0.5 border border-amber-500/40">
                          {tok.text}
                        </mark>
                      );
                    }
                    if (tok.category === "roleplay") {
                      return (
                        <mark key={i} className="bg-purple-500/25 text-purple-300 font-bold px-1 rounded mx-0.5 border border-purple-500/40">
                          {tok.text}
                        </mark>
                      );
                    }
                    if (tok.category === "security") {
                      return (
                        <mark key={i} className="bg-cyan-500/25 text-cyan-300 font-bold px-1 rounded mx-0.5 border border-cyan-500/40">
                          {tok.text}
                        </mark>
                      );
                    }
                    return <span key={i}>{tok.text}</span>;
                  })}
                </p>
              </div>
            )}

            {/* Textarea */}
            <div className="flex-1 min-h-[140px] flex flex-col">
              <textarea
                value={promptText}
                onChange={(e) => {
                  setPromptText(e.target.value);
                  setSelectedPresetId("");
                }}
                placeholder="Enter an LLM prompt or attack payload to analyze in real time..."
                className="w-full flex-1 p-3.5 rounded-xl glass-input resize-none font-mono text-xs sm:text-sm leading-relaxed focus:ring-1"
                rows={5}
              />
            </div>

            {/* Bottom bar */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mt-4 pt-3 border-t border-[var(--border-subtle)]">
              <div className="flex items-center gap-3 text-xs text-[var(--text-muted)] font-mono">
                <span>{promptText.length} chars</span>
                <span>•</span>
                <span>{promptText.trim() ? promptText.trim().split(/\s+/).length : 0} tokens</span>
                <span>•</span>
                <span>{currentResult?.telemetry.inference_time_ms || 14}ms latency</span>
              </div>

              <button
                onClick={() => handleAnalyze(promptText)}
                disabled={isAnalyzing || !promptText.trim()}
                className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-medium text-xs sm:text-sm shadow-md shadow-blue-500/25 flex items-center justify-center gap-2 transition-all active:scale-95 disabled:opacity-50"
              >
                {isAnalyzing ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span>Evaluating...</span>
                  </>
                ) : (
                  <>
                    <Zap className="w-4 h-4" />
                    <span>Run ML Pipeline</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Neural Verdict & Softmax Probabilities (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Main Verdict Card */}
          <div
            className={`glass-panel rounded-2xl p-5 border transition-all duration-300 relative overflow-hidden ${
              isMalicious
                ? "border-red-500/40 bg-gradient-to-b from-red-500/10 via-transparent to-transparent glow-crimson"
                : "border-emerald-500/40 bg-gradient-to-b from-emerald-500/10 via-transparent to-transparent glow-emerald"
            }`}
          >
            <div className="flex items-start justify-between">
              <div>
                <span className="text-[11px] font-bold uppercase tracking-wider text-[var(--text-secondary)]">
                  Neural Decision Verdict
                </span>
                <div className="flex items-center gap-2 mt-1">
                  {isMalicious ? (
                    <ShieldAlert className="w-7 h-7 text-red-500" />
                  ) : (
                    <ShieldCheck className="w-7 h-7 text-emerald-500" />
                  )}
                  <h3 className={`text-2xl font-bold tracking-tight ${isMalicious ? "text-red-400" : "text-emerald-400"}`}>
                    {isMalicious ? "Malicious Attack" : "Safe Prompt"}
                  </h3>
                </div>
              </div>

              {/* Confidence Badge */}
              <div className="text-right">
                <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider block">
                  Confidence
                </span>
                <span className="text-xl font-bold font-mono text-[var(--text-primary)]">
                  {Math.round((currentResult?.confidence || 0.85) * 100)}%
                </span>
              </div>
            </div>

            <p className="text-xs text-[var(--text-secondary)] mt-3 leading-relaxed">
              {isMalicious
                ? "High probability of prompt injection, jailbreak framing, or instruction override pattern."
                : "Standard query patterns detected. No anomalous behavioral or semantic jailbreak triggers identified."}
            </p>

            {/* Radial Meters: Dual Probability Split */}
            <div className="grid grid-cols-2 gap-4 mt-5 pt-4 border-t border-[var(--border-subtle)]">
              {/* Safe Gauge */}
              <div className="flex items-center gap-3">
                <div className="relative w-16 h-16 flex items-center justify-center">
                  <svg className="w-16 h-16 transform -rotate-90">
                    <circle
                      cx="32"
                      cy="32"
                      r="26"
                      stroke="currentColor"
                      strokeWidth="5"
                      className="text-black/10 dark:text-white/10"
                      fill="transparent"
                    />
                    <circle
                      cx="32"
                      cy="32"
                      r="26"
                      stroke="#30d158"
                      strokeWidth="5"
                      strokeDasharray={2 * Math.PI * 26}
                      strokeDashoffset={2 * Math.PI * 26 * (1 - safeProb / 100)}
                      strokeLinecap="round"
                      fill="transparent"
                      className="transition-all duration-500"
                    />
                  </svg>
                  <span className="absolute text-xs font-bold font-mono text-emerald-400">
                    {safeProb}%
                  </span>
                </div>
                <div>
                  <span className="text-xs font-semibold block">P(Safe)</span>
                  <span className="text-[10px] text-[var(--text-muted)]">Benign</span>
                </div>
              </div>

              {/* Malicious Gauge */}
              <div className="flex items-center gap-3">
                <div className="relative w-16 h-16 flex items-center justify-center">
                  <svg className="w-16 h-16 transform -rotate-90">
                    <circle
                      cx="32"
                      cy="32"
                      r="26"
                      stroke="currentColor"
                      strokeWidth="5"
                      className="text-black/10 dark:text-white/10"
                      fill="transparent"
                    />
                    <circle
                      cx="32"
                      cy="32"
                      r="26"
                      stroke="#ff453a"
                      strokeWidth="5"
                      strokeDasharray={2 * Math.PI * 26}
                      strokeDashoffset={2 * Math.PI * 26 * (1 - maliciousProb / 100)}
                      strokeLinecap="round"
                      fill="transparent"
                      className="transition-all duration-500"
                    />
                  </svg>
                  <span className="absolute text-xs font-bold font-mono text-red-400">
                    {maliciousProb}%
                  </span>
                </div>
                <div>
                  <span className="text-xs font-semibold block">P(Malicious)</span>
                  <span className="text-[10px] text-[var(--text-muted)]">Threat</span>
                </div>
              </div>
            </div>

            {/* Quick deep dive actions */}
            <div className="grid grid-cols-2 gap-2 mt-4 pt-3 border-t border-[var(--border-subtle)]">
              {onSelectFeatureTab && (
                <button
                  onClick={onSelectFeatureTab}
                  className="px-3 py-2 rounded-xl text-xs font-medium bg-black/5 dark:bg-white/5 hover:bg-black/10 dark:hover:bg-white/10 transition-colors flex items-center justify-between text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                >
                  <span>15-Feature Radar</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              )}
              {onSelectArchitectureTab && (
                <button
                  onClick={onSelectArchitectureTab}
                  className="px-3 py-2 rounded-xl text-xs font-medium bg-black/5 dark:bg-white/5 hover:bg-black/10 dark:hover:bg-white/10 transition-colors flex items-center justify-between text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                >
                  <span>Neural Flow</span>
                  <Layers className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          </div>

          {/* Telemetry metadata snippet */}
          <div className="glass-panel rounded-2xl p-4 text-xs space-y-2">
            <div className="flex items-center justify-between text-[var(--text-secondary)]">
              <span className="flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-blue-400" />
                <span>Backend Engine:</span>
              </span>
              <span className="font-mono text-[var(--text-primary)]">
                {currentResult?.telemetry.embedder_backend}
              </span>
            </div>
            <div className="flex items-center justify-between text-[var(--text-secondary)]">
              <span className="flex items-center gap-1.5">
                <Code2 className="w-3.5 h-3.5 text-indigo-400" />
                <span>Input Dimensions:</span>
              </span>
              <span className="font-mono text-[var(--text-primary)]">
                15 (Beh) + 2000 (TFIDF) + 768 (Emb) = 2,783 dims
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
