import { PredictionResult } from "@/types/ml";
import { runClientInference, extractBehavioralFeatures, highlightTokens } from "./ml-engine";

const FASTAPI_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface BackendHealth {
  status: "connected" | "disconnected";
  embedder_backend?: string;
  url: string;
}

export async function checkBackendHealth(): Promise<BackendHealth> {
  try {
    const res = await fetch(`${FASTAPI_BASE_URL}/health`, {
      method: "GET",
      signal: AbortSignal.timeout(1500),
    });
    if (res.ok) {
      const data = await res.json();
      return {
        status: "connected",
        embedder_backend: data.embedder_backend || "DeBERTa / Hashing",
        url: FASTAPI_BASE_URL,
      };
    }
  } catch {
    // Expected when FastAPI is not running locally
  }
  return {
    status: "disconnected",
    embedder_backend: "Client Sim Engine",
    url: FASTAPI_BASE_URL,
  };
}

export async function classifyPrompt(text: string): Promise<PredictionResult> {
  const startTime = performance.now();
  try {
    const res = await fetch(`${FASTAPI_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      signal: AbortSignal.timeout(3000),
    });

    if (res.ok) {
      const data = await res.json();
      const elapsed = Math.round(performance.now() - startTime);
      const behavioral = extractBehavioralFeatures(text);
      const highlights = highlightTokens(text);

      return {
        prompt: data.prompt || text,
        prediction: data.prediction === "Malicious" ? "Malicious" : "Safe",
        confidence: data.confidence,
        probabilities: {
          Safe: data.probabilities?.Safe ?? (data.prediction === "Safe" ? data.confidence : 1 - data.confidence),
          Malicious: data.probabilities?.Malicious ?? (data.prediction === "Malicious" ? data.confidence : 1 - data.confidence),
        },
        telemetry: {
          behavioral_features: behavioral,
          feature_dims: {
            behavioral_dim: 15,
            tfidf_dim: 2000,
            embedding_dim: 768,
            fused_dim: 2783,
          },
          embedder_backend: "FastAPI Live Artifacts",
          execution_device: "PyTorch (CPU/CUDA)",
          inference_time_ms: elapsed,
          token_highlights: highlights,
        },
        source: "fastapi",
      };
    }
  } catch {
    // Graceful fallback to client engine
  }

  return runClientInference(text);
}
