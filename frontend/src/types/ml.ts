export type PredictionClass = "Safe" | "Malicious";

export interface BehavioralFeatureValues {
  prompt_length: number;
  token_count: number;
  uppercase_ratio: number;
  digit_ratio: number;
  punctuation_ratio: number;
  special_char_count: number;
  keyword_frequency: number;
  roleplay_indicator_count: number;
  instruction_override_count: number;
  jailbreak_keyword_count: number;
  entropy: number;
  repetition_score: number;
  url_count: number;
  code_block_count: number;
  markdown_count: number;
}

export interface HighlightedToken {
  text: string;
  category: "override" | "jailbreak" | "roleplay" | "security" | "normal";
}

export interface FeatureDimensionInfo {
  behavioral_dim: number;
  tfidf_dim: number;
  embedding_dim: number;
  fused_dim: number;
}

export interface MLTelemetry {
  behavioral_features: BehavioralFeatureValues;
  feature_dims: FeatureDimensionInfo;
  embedder_backend: string;
  execution_device: string;
  inference_time_ms: number;
  token_highlights: HighlightedToken[];
  top_tfidf_tokens?: { token: string; weight: number }[];
  layer_activations?: {
    fused_norm: number;
    hidden_1_norm: number;
    hidden_2_norm: number;
  };
}

export interface PredictionResult {
  prompt: string;
  prediction: PredictionClass;
  confidence: number;
  probabilities: {
    Safe: number;
    Malicious: number;
  };
  telemetry: MLTelemetry;
  source: "fastapi" | "client_engine";
}

export interface AttackPreset {
  id: string;
  title: string;
  category: "Jailbreak" | "Prompt Injection" | "Roleplay" | "Obfuscation" | "Benign";
  description: string;
  prompt: string;
  expectedClass: PredictionClass;
}

export interface EvaluationMetricData {
  total_samples: number;
  safe_samples: number;
  malicious_samples: number;
  train_split: number;
  test_split: number;
  accuracy: number;
  roc_auc: number;
  precision: number;
  recall: number;
  f1_score: number;
  confusion_matrix: {
    true_negative: number;
    false_positive: number;
    false_negative: number;
    true_positive: number;
  };
}
