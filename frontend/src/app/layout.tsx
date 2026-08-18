import type { Metadata } from "next";
import { ThemeProvider } from "@/context/ThemeContext";
import "./globals.css";

export const metadata: Metadata = {
  title: "BehaveGuard ML Studio — Hybrid Prompt Injection Defense",
  description:
    "Interactive Machine Learning Dashboard showcasing Tri-Modal Neural Fusion, 15-Feature Behavioral Explainability, and Jailbreak Detection.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning className="h-full">
      <body className="min-h-full flex flex-col antialiased selection:bg-blue-500/20 selection:text-blue-400">
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
