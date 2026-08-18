"use client";

import React, { useState } from "react";
import { Navbar, NavTab } from "@/components/Navbar";
import { InferencePlayground } from "@/components/InferencePlayground";
import { ArchitectureFlow } from "@/components/ArchitectureFlow";
import { FeatureRadar } from "@/components/FeatureRadar";
import { ModelEvaluation } from "@/components/ModelEvaluation";
import { BatchAudit } from "@/components/BatchAudit";
import { PredictionResult } from "@/types/ml";
import { ATTACK_PRESETS, runClientInference } from "@/lib/ml-engine";
import { ShieldCheck, Cpu, Database, Network } from "lucide-react";

export default function Home() {
  const [activeTab, setActiveTab] = useState<NavTab>("playground");
  const [currentResult, setCurrentResult] = useState<PredictionResult>(() =>
    runClientInference(ATTACK_PRESETS[0].prompt)
  );

  return (
    <div className="min-h-screen flex flex-col transition-colors duration-300">
      {/* Floating Apple Navbar */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 space-y-6">
        {/* ML Studio Overview Strip */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="p-3 rounded-2xl glass-panel flex items-center gap-3 border border-[var(--border-subtle)]">
            <div className="w-8 h-8 rounded-xl bg-blue-500/15 flex items-center justify-center text-blue-400">
              <Network className="w-4 h-4" />
            </div>
            <div>
              <span className="text-[10px] text-[var(--text-muted)] font-mono uppercase block">
                Model Architecture
              </span>
              <span className="text-xs font-bold text-[var(--text-primary)]">
                Tri-Modal Fusion Net
              </span>
            </div>
          </div>

          <div className="p-3 rounded-2xl glass-panel flex items-center gap-3 border border-[var(--border-subtle)]">
            <div className="w-8 h-8 rounded-xl bg-indigo-500/15 flex items-center justify-center text-indigo-400">
              <Cpu className="w-4 h-4" />
            </div>
            <div>
              <span className="text-[10px] text-[var(--text-muted)] font-mono uppercase block">
                Feature Space
              </span>
              <span className="text-xs font-bold text-[var(--text-primary)] font-mono">
                2,783 Dimensions
              </span>
            </div>
          </div>

          <div className="p-3 rounded-2xl glass-panel flex items-center gap-3 border border-[var(--border-subtle)]">
            <div className="w-8 h-8 rounded-xl bg-emerald-500/15 flex items-center justify-center text-emerald-400">
              <ShieldCheck className="w-4 h-4" />
            </div>
            <div>
              <span className="text-[10px] text-[var(--text-muted)] font-mono uppercase block">
                Malicious Recall
              </span>
              <span className="text-xs font-bold text-emerald-400 font-mono">
                83.3% Test Catch
              </span>
            </div>
          </div>

          <div className="p-3 rounded-2xl glass-panel flex items-center gap-3 border border-[var(--border-subtle)]">
            <div className="w-8 h-8 rounded-xl bg-purple-500/15 flex items-center justify-center text-purple-400">
              <Database className="w-4 h-4" />
            </div>
            <div>
              <span className="text-[10px] text-[var(--text-muted)] font-mono uppercase block">
                Behavioral Signals
              </span>
              <span className="text-xs font-bold text-[var(--text-primary)] font-mono">
                15 Domain Features
              </span>
            </div>
          </div>
        </div>

        {/* Dynamic Tab Views */}
        <div className="transition-all duration-300">
          {activeTab === "playground" && (
            <InferencePlayground
              onSelectFeatureTab={() => setActiveTab("explainability")}
              onSelectArchitectureTab={() => setActiveTab("architecture")}
              currentResult={currentResult}
              setCurrentResult={setCurrentResult}
            />
          )}

          {activeTab === "architecture" && <ArchitectureFlow />}

          {activeTab === "explainability" && (
            <FeatureRadar currentResult={currentResult} />
          )}

          {activeTab === "benchmarks" && <ModelEvaluation />}

          {activeTab === "batch" && <BatchAudit />}
        </div>
      </main>

      {/* Apple-style minimalist footer */}
      <footer className="border-t border-[var(--border-subtle)] py-6 text-center text-xs text-[var(--text-muted)] mt-auto">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>BehaveGuard — Hybrid Prompt Injection & Jailbreak Defense System</span>
          <span className="font-mono text-[11px]">
            PyTorch • DeBERTa-v3 • TF-IDF • FastAPI • Next.js
          </span>
        </div>
      </footer>
    </div>
  );
}
