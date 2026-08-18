"use client";

import React, { useEffect, useState } from "react";
import { useTheme } from "@/context/ThemeContext";
import {
  Sun,
  Moon,
  Activity,
  Cpu,
  Layers,
  Radar,
  BarChart3,
  ListChecks,
  Sparkles,
  Server,
} from "lucide-react";
import { checkBackendHealth, BackendHealth } from "@/lib/api-client";

export type NavTab = "playground" | "architecture" | "explainability" | "benchmarks" | "batch";

interface NavbarProps {
  activeTab: NavTab;
  setActiveTab: (tab: NavTab) => void;
}

export function Navbar({ activeTab, setActiveTab }: NavbarProps) {
  const { theme, toggleTheme } = useTheme();
  const [health, setHealth] = useState<BackendHealth>({
    status: "disconnected",
    embedder_backend: "Client Engine",
    url: "http://localhost:8000",
  });

  useEffect(() => {
    const fetchHealth = async () => {
      const h = await checkBackendHealth();
      setHealth(h);
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 8000);
    return () => clearInterval(interval);
  }, []);

  const navItems: { id: NavTab; label: string; icon: React.ReactNode }[] = [
    { id: "playground", label: "Inference Playground", icon: <Activity className="w-4 h-4" /> },
    { id: "architecture", label: "Tri-Modal Architecture", icon: <Layers className="w-4 h-4" /> },
    { id: "explainability", label: "XAI Radar", icon: <Radar className="w-4 h-4" /> },
    { id: "benchmarks", label: "Model Metrics & ROC", icon: <BarChart3 className="w-4 h-4" /> },
    { id: "batch", label: "Batch Audit", icon: <ListChecks className="w-4 h-4" /> },
  ];

  return (
    <header className="sticky top-0 z-50 w-full glass-nav transition-colors duration-300">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand & System Tag */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-500 to-cyan-400 flex items-center justify-center shadow-md shadow-blue-500/20">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-semibold tracking-tight text-base sm:text-lg">
                BehaveGuard
              </span>
              <span className="text-[10px] uppercase font-bold tracking-widest px-2 py-0.5 rounded-full bg-blue-500/15 text-blue-500 dark:text-blue-400 border border-blue-500/20">
                ML Studio
              </span>
            </div>
            <p className="text-[11px] text-[var(--text-secondary)] hidden sm:block">
              Hybrid Neural Prompt Injection & Jailbreak Defense
            </p>
          </div>
        </div>

        {/* Center Segmented Navigation Bar */}
        <nav className="hidden md:flex items-center gap-1 p-1 rounded-full bg-[rgba(120,120,128,0.12)] border border-[var(--border-subtle)] backdrop-blur-md">
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-medium transition-all duration-200 ${
                  isActive
                    ? "bg-white dark:bg-white/15 text-black dark:text-white shadow-sm font-semibold"
                    : "text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/5 dark:hover:bg-white/5"
                }`}
              >
                {item.icon}
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Right Controls: Backend Status & Theme Switch */}
        <div className="flex items-center gap-3">
          {/* Backend Status Pill */}
          <div
            title={
              health.status === "connected"
                ? `Connected to FastAPI (${health.url})`
                : "FastAPI server offline — Using in-browser WebAssembly ML Engine"
            }
            className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs glass-panel border border-[var(--border-subtle)]"
          >
            <span className="relative flex h-2 w-2">
              {health.status === "connected" ? (
                <>
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </>
              ) : (
                <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
              )}
            </span>
            <span className="hidden sm:inline text-[11px] font-medium text-[var(--text-secondary)]">
              {health.status === "connected" ? "FastAPI Live" : "Client Engine"}
            </span>
          </div>

          {/* Dark/Light Mode Switch */}
          <button
            onClick={toggleTheme}
            aria-label="Toggle theme"
            className="w-9 h-9 rounded-full glass-panel flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors hover:scale-105 active:scale-95"
          >
            {theme === "dark" ? (
              <Sun className="w-4 h-4 text-amber-400 transition-transform duration-300 rotate-0 hover:rotate-45" />
            ) : (
              <Moon className="w-4 h-4 text-indigo-500 transition-transform duration-300 rotate-0 hover:-rotate-12" />
            )}
          </button>
        </div>
      </div>

      {/* Mobile Tab Navigation */}
      <div className="flex md:hidden overflow-x-auto px-4 py-2 border-t border-[var(--border-subtle)] gap-2 scrollbar-none">
        {navItems.map((item) => {
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs whitespace-nowrap transition-colors ${
                isActive
                  ? "bg-blue-600 text-white font-medium shadow-sm"
                  : "bg-black/5 dark:bg-white/5 text-[var(--text-secondary)]"
              }`}
            >
              {item.icon}
              <span>{item.label}</span>
            </button>
          );
        })}
      </div>
    </header>
  );
}
