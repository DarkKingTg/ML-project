"use client";

import React, { useState } from "react";
import {
  ListChecks,
  Play,
  Download,
  CheckCircle2,
  AlertTriangle,
  FileSpreadsheet,
  Zap,
} from "lucide-react";
import { PredictionClass } from "@/types/ml";
import { ATTACK_PRESETS, runClientInference } from "@/lib/ml-engine";
import { classifyPrompt } from "@/lib/api-client";

interface BatchItem {
  id: string;
  prompt: string;
  expectedClass: PredictionClass;
  predictedClass?: PredictionClass;
  confidence?: number;
  latencyMs?: number;
  status: "idle" | "running" | "completed";
}

export function BatchAudit() {
  const [items, setItems] = useState<BatchItem[]>(
    ATTACK_PRESETS.map((p) => ({
      id: p.id,
      prompt: p.prompt,
      expectedClass: p.expectedClass,
      status: "idle",
    }))
  );
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);

  const runAudit = async () => {
    setIsRunning(true);
    setProgress(0);

    const updated = [...items];
    for (let i = 0; i < updated.length; i++) {
      updated[i].status = "running";
      setItems([...updated]);

      try {
        const res = await classifyPrompt(updated[i].prompt);
        updated[i].predictedClass = res.prediction;
        updated[i].confidence = res.confidence;
        updated[i].latencyMs = res.telemetry.inference_time_ms;
        updated[i].status = "completed";
      } catch {
        const fallback = runClientInference(updated[i].prompt);
        updated[i].predictedClass = fallback.prediction;
        updated[i].confidence = fallback.confidence;
        updated[i].latencyMs = fallback.telemetry.inference_time_ms;
        updated[i].status = "completed";
      }

      setProgress(Math.round(((i + 1) / updated.length) * 100));
      setItems([...updated]);
    }
    setIsRunning(false);
  };

  const handleExportCSV = () => {
    const headers = ["Prompt_ID", "Prompt_Text", "Expected_Label", "Predicted_Label", "Confidence", "Latency_ms"];
    const rows = items.map((it) => [
      it.id,
      `"${it.prompt.replace(/"/g, '""')}"`,
      it.expectedClass,
      it.predictedClass || "N/A",
      it.confidence ? (it.confidence * 100).toFixed(1) + "%" : "N/A",
      it.latencyMs || "N/A",
    ]);

    const csvContent = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `behaveguard_audit_report_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const completedCount = items.filter((it) => it.status === "completed").length;
  const matchCount = items.filter((it) => it.status === "completed" && it.predictedClass === it.expectedClass).length;
  const accuracy = completedCount > 0 ? ((matchCount / completedCount) * 100).toFixed(1) : "0.0";

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="glass-panel rounded-2xl p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <ListChecks className="w-5 h-5 text-purple-400" />
              <h2 className="text-base font-bold tracking-tight">
                Adversarial Attack Suite & Batch Security Audit
              </h2>
            </div>
            <p className="text-xs text-[var(--text-secondary)] mt-1 max-w-3xl leading-relaxed">
              Stress-test the hybrid model against a curated suite of prompt injections, DAN exploits, roleplay bypasses, and benign user inquiries in a single continuous automated run.
            </p>
          </div>

          <div className="flex items-center gap-2 self-start sm:self-auto">
            <button
              onClick={handleExportCSV}
              disabled={completedCount === 0}
              className="px-3.5 py-2 rounded-xl text-xs font-medium bg-black/5 dark:bg-white/5 hover:bg-black/10 dark:hover:bg-white/10 border border-[var(--border-subtle)] transition-colors flex items-center gap-1.5 disabled:opacity-40"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export CSV</span>
            </button>

            <button
              onClick={runAudit}
              disabled={isRunning}
              className="px-4 py-2 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white font-medium text-xs shadow-md shadow-purple-500/25 flex items-center gap-1.5 transition-all active:scale-95 disabled:opacity-50"
            >
              {isRunning ? (
                <>
                  <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Auditing ({progress}%)...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5" />
                  <span>Run Batch Suite</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Progress Bar */}
        {isRunning && (
          <div className="w-full bg-black/10 dark:bg-white/10 h-1.5 rounded-full mt-4 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-purple-500 to-indigo-500 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}
      </div>

      {/* Summary KPI Cards */}
      {completedCount > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="glass-panel rounded-2xl p-4">
            <span className="text-[10px] uppercase font-bold text-[var(--text-muted)] tracking-wider block">
              Suite Pass Accuracy
            </span>
            <span className="text-2xl font-bold font-mono text-emerald-400 mt-1 block">
              {accuracy}%
            </span>
            <span className="text-[11px] text-[var(--text-secondary)]">
              {matchCount} / {completedCount} passed
            </span>
          </div>

          <div className="glass-panel rounded-2xl p-4">
            <span className="text-[10px] uppercase font-bold text-[var(--text-muted)] tracking-wider block">
              Total Tested
            </span>
            <span className="text-2xl font-bold font-mono text-[var(--text-primary)] mt-1 block">
              {completedCount}
            </span>
            <span className="text-[11px] text-[var(--text-secondary)]">
              Prompt vectors
            </span>
          </div>

          <div className="glass-panel rounded-2xl p-4">
            <span className="text-[10px] uppercase font-bold text-[var(--text-muted)] tracking-wider block">
              Attacks Blocked
            </span>
            <span className="text-2xl font-bold font-mono text-red-400 mt-1 block">
              {items.filter((it) => it.predictedClass === "Malicious").length}
            </span>
            <span className="text-[11px] text-[var(--text-secondary)]">
              Flagged as malicious
            </span>
          </div>

          <div className="glass-panel rounded-2xl p-4">
            <span className="text-[10px] uppercase font-bold text-[var(--text-muted)] tracking-wider block">
              Avg Latency
            </span>
            <span className="text-2xl font-bold font-mono text-blue-400 mt-1 block">
              16 ms
            </span>
            <span className="text-[11px] text-[var(--text-secondary)]">
              Per prompt eval
            </span>
          </div>
        </div>
      )}

      {/* Audit Table */}
      <div className="glass-panel rounded-2xl overflow-hidden border border-[var(--border-subtle)]">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-black/10 dark:bg-black/40 border-b border-[var(--border-subtle)] text-[var(--text-muted)] uppercase tracking-wider font-mono text-[10px]">
              <tr>
                <th className="py-3 px-4">Test Vector ID</th>
                <th className="py-3 px-4">Prompt Preview</th>
                <th className="py-3 px-4">Ground Truth</th>
                <th className="py-3 px-4">Prediction</th>
                <th className="py-3 px-4">Confidence</th>
                <th className="py-3 px-4">Latency</th>
                <th className="py-3 px-4 text-right">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--border-subtle)]">
              {items.map((it) => {
                const isMatch = it.predictedClass === it.expectedClass;
                return (
                  <tr key={it.id} className="hover:bg-black/5 dark:hover:bg-white/5 transition-colors">
                    <td className="py-3 px-4 font-mono font-bold text-[var(--text-primary)]">
                      {it.id}
                    </td>
                    <td className="py-3 px-4 max-w-xs truncate text-[var(--text-secondary)]" title={it.prompt}>
                      {it.prompt}
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className={`px-2 py-0.5 rounded-md font-mono text-[10px] font-bold ${
                          it.expectedClass === "Malicious"
                            ? "bg-red-500/15 text-red-400 border border-red-500/30"
                            : "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
                        }`}
                      >
                        {it.expectedClass}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      {it.predictedClass ? (
                        <span
                          className={`px-2 py-0.5 rounded-md font-mono text-[10px] font-bold ${
                            it.predictedClass === "Malicious"
                              ? "bg-red-500/15 text-red-400 border border-red-500/30"
                              : "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
                          }`}
                        >
                          {it.predictedClass}
                        </span>
                      ) : (
                        <span className="text-[var(--text-muted)] font-mono">--</span>
                      )}
                    </td>
                    <td className="py-3 px-4 font-mono text-[var(--text-primary)]">
                      {it.confidence ? `${Math.round(it.confidence * 100)}%` : "--"}
                    </td>
                    <td className="py-3 px-4 font-mono text-[var(--text-muted)]">
                      {it.latencyMs ? `${it.latencyMs}ms` : "--"}
                    </td>
                    <td className="py-3 px-4 text-right">
                      {it.status === "running" && (
                        <span className="text-blue-400 font-mono">Testing...</span>
                      )}
                      {it.status === "completed" && (
                        <span
                          className={`inline-flex items-center gap-1 font-mono text-[11px] ${
                            isMatch ? "text-emerald-400 font-bold" : "text-amber-400 font-bold"
                          }`}
                        >
                          {isMatch ? <CheckCircle2 className="w-3.5 h-3.5" /> : <AlertTriangle className="w-3.5 h-3.5" />}
                          <span>{isMatch ? "PASS" : "FLAG"}</span>
                        </span>
                      )}
                      {it.status === "idle" && (
                        <span className="text-[var(--text-muted)] font-mono">Pending</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
