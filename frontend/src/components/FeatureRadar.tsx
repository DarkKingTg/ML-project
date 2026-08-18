"use client";

import React, { useState } from "react";
import {
  Radar as RadarIcon,
  ShieldAlert,
  Sliders,
  HelpCircle,
  Activity,
  Zap,
} from "lucide-react";
import { PredictionResult, BehavioralFeatureValues } from "@/types/ml";
import { FEATURE_METADATA } from "@/lib/ml-engine";

interface FeatureRadarProps {
  currentResult: PredictionResult | null;
}

export function FeatureRadar({ currentResult }: FeatureRadarProps) {
  const [selectedCategory, setSelectedCategory] = useState<"All" | "Security" | "Structural" | "Surface">("All");

  const features = currentResult?.telemetry.behavioral_features || {
    prompt_length: 120,
    token_count: 22,
    uppercase_ratio: 0.08,
    digit_ratio: 0.02,
    punctuation_ratio: 0.06,
    special_char_count: 1,
    keyword_frequency: 2,
    roleplay_indicator_count: 1,
    instruction_override_count: 1,
    jailbreak_keyword_count: 1,
    entropy: 4.2,
    repetition_score: 0.12,
    url_count: 0,
    code_block_count: 0,
    markdown_count: 1,
  };

  // Normalization logic for the 15-axis radar chart
  const normalize = (key: keyof BehavioralFeatureValues, val: number): number => {
    switch (key) {
      case "prompt_length":
        return Math.min(1, val / 400);
      case "token_count":
        return Math.min(1, val / 60);
      case "uppercase_ratio":
        return Math.min(1, val / 0.5);
      case "digit_ratio":
        return Math.min(1, val / 0.3);
      case "punctuation_ratio":
        return Math.min(1, val / 0.2);
      case "special_char_count":
        return Math.min(1, val / 8);
      case "keyword_frequency":
        return Math.min(1, val / 4);
      case "roleplay_indicator_count":
        return Math.min(1, val / 2);
      case "instruction_override_count":
        return Math.min(1, val / 2);
      case "jailbreak_keyword_count":
        return Math.min(1, val / 2);
      case "entropy":
        return Math.min(1, Math.max(0, (val - 2.5) / 2.5));
      case "repetition_score":
        return Math.min(1, val / 0.5);
      case "url_count":
        return Math.min(1, val / 2);
      case "code_block_count":
        return Math.min(1, val / 2);
      case "markdown_count":
        return Math.min(1, val / 4);
      default:
        return 0.5;
    }
  };

  // Generate SVG Radar Polygon Points
  const center = 150;
  const maxRadius = 110;
  const numAxes = FEATURE_METADATA.length;
  const angleStep = (2 * Math.PI) / numAxes;

  const polygonPoints = FEATURE_METADATA.map((meta, i) => {
    const norm = normalize(meta.key, features[meta.key]);
    const radius = norm * maxRadius;
    const angle = i * angleStep - Math.PI / 2;
    const x = center + radius * Math.cos(angle);
    const y = center + radius * Math.sin(angle);
    return `${x},${y}`;
  }).join(" ");

  const filteredFeatures = FEATURE_METADATA.filter((meta) => {
    if (selectedCategory === "All") return true;
    return meta.category === selectedCategory;
  });

  return (
    <div className="space-y-6">
      {/* Header Info */}
      <div className="glass-panel rounded-2xl p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <RadarIcon className="w-5 h-5 text-indigo-400" />
              <h2 className="text-base font-bold tracking-tight">
                15-Feature Behavioral Explainability Radar (XAI)
              </h2>
            </div>
            <p className="text-xs text-[var(--text-secondary)] mt-1 max-w-3xl leading-relaxed">
              Unlike black-box models, BehaveGuard extracts 15 domain-engineered behavioral signals from raw text. This radar maps statistical surface traits and security lexicon activations in real time.
            </p>
          </div>

          {/* Category Filter Pills */}
          <div className="flex items-center gap-1.5 p-1 rounded-xl bg-black/5 dark:bg-white/5 border border-[var(--border-subtle)] self-start sm:self-auto">
            {(["All", "Security", "Structural", "Surface"] as const).map((cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors ${
                  selectedCategory === cat
                    ? "bg-blue-600 text-white font-semibold shadow-sm"
                    : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Grid: Left Radar Chart & Right Feature Details Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: SVG Multi-Axis Spider Chart (5 Cols) */}
        <div className="lg:col-span-5 glass-panel rounded-2xl p-5 flex flex-col items-center justify-center relative">
          <div className="w-full flex items-center justify-between border-b border-[var(--border-subtle)] pb-3 mb-2">
            <span className="text-xs font-bold uppercase tracking-wider text-[var(--text-secondary)]">
              Behavioral Fingerprint
            </span>
            <span className="text-[11px] font-mono text-indigo-400">15 Axes</span>
          </div>

          {/* SVG Radar Visual */}
          <div className="relative w-full max-w-[320px] aspect-square flex items-center justify-center my-2">
            <svg viewBox="0 0 300 300" className="w-full h-full">
              {/* Concentric Guide Circles */}
              {[0.25, 0.5, 0.75, 1.0].map((ratio, idx) => (
                <circle
                  key={idx}
                  cx={center}
                  cy={center}
                  r={maxRadius * ratio}
                  fill="transparent"
                  stroke="currentColor"
                  strokeWidth="1"
                  className="text-black/5 dark:text-white/10"
                />
              ))}

              {/* Axis Spoke Lines */}
              {FEATURE_METADATA.map((_, i) => {
                const angle = i * angleStep - Math.PI / 2;
                const x = center + maxRadius * Math.cos(angle);
                const y = center + maxRadius * Math.sin(angle);
                return (
                  <line
                    key={i}
                    x1={center}
                    y1={center}
                    x2={x}
                    y2={y}
                    stroke="currentColor"
                    strokeWidth="1"
                    className="text-black/5 dark:text-white/10"
                  />
                );
              })}

              {/* Filled Radar Polygon */}
              <polygon
                points={polygonPoints}
                className="fill-indigo-500/25 dark:fill-indigo-500/35 stroke-indigo-400 dark:stroke-indigo-400 transition-all duration-300"
                strokeWidth="2"
              />

              {/* Individual Vertex Nodes */}
              {FEATURE_METADATA.map((meta, i) => {
                const norm = normalize(meta.key, features[meta.key]);
                const radius = norm * maxRadius;
                const angle = i * angleStep - Math.PI / 2;
                const x = center + radius * Math.cos(angle);
                const y = center + radius * Math.sin(angle);
                const isTriggered = norm >= 0.5;

                return (
                  <circle
                    key={i}
                    cx={x}
                    cy={y}
                    r={isTriggered ? 3.5 : 2.5}
                    className={`${
                      isTriggered
                        ? "fill-red-400 stroke-white dark:stroke-black"
                        : "fill-indigo-400"
                    } transition-all duration-300`}
                    strokeWidth="1"
                  />
                );
              })}
            </svg>
          </div>

          <div className="text-center text-[11px] text-[var(--text-muted)] mt-2">
            Higher values extend toward the outer perimeter. Red nodes indicate elevated risk triggers.
          </div>
        </div>

        {/* Right Column: Detailed Metric Cards (7 Cols) */}
        <div className="lg:col-span-7 space-y-3">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-h-[460px] overflow-y-auto pr-1">
            {filteredFeatures.map((meta) => {
              const val = features[meta.key];
              const norm = normalize(meta.key, val);
              const isHigh = norm >= 0.5;

              return (
                <div
                  key={meta.key}
                  className={`p-3.5 rounded-xl border transition-all glass-panel ${
                    isHigh
                      ? "border-red-500/30 bg-red-500/5"
                      : "border-[var(--border-subtle)]"
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <span className="text-[10px] font-mono uppercase tracking-wider text-[var(--text-muted)] block">
                        {meta.category}
                      </span>
                      <span className="text-xs font-semibold text-[var(--text-primary)]">
                        {meta.label}
                      </span>
                    </div>
                    <span
                      className={`text-xs font-bold font-mono px-2 py-0.5 rounded-md ${
                        isHigh
                          ? "bg-red-500/20 text-red-400 border border-red-500/30"
                          : "bg-black/5 dark:bg-white/5 text-[var(--text-secondary)]"
                      }`}
                    >
                      {typeof val === "number" && !Number.isInteger(val) ? val.toFixed(3) : val}
                    </span>
                  </div>

                  <p className="text-[11px] text-[var(--text-secondary)] mt-1.5 line-clamp-2">
                    {meta.description}
                  </p>

                  {/* Progress Bar */}
                  <div className="w-full bg-black/10 dark:bg-white/10 h-1.5 rounded-full mt-3 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-300 ${
                        isHigh ? "bg-red-500" : "bg-indigo-500"
                      }`}
                      style={{ width: `${Math.round(norm * 100)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
