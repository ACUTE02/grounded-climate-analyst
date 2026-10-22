// Shapes the backend sends (backend/api/schemas.py). Nothing here is computed in the
// browser: labels, scores and flags are shown exactly as received.

export interface Grounding {
  status: "clean" | "warning" | "not_generated" | string;
  grounded: boolean;
  total_checked: number;
  unverified_numbers: string[];
  report_missing: boolean;
  explanation: string;
}

export interface HorizonConfidence {
  horizon: string;
  skill_score: number;
  method: string;
  label: string;
  reliable: boolean;
}

export interface MissingData {
  what: string;
  available: boolean;
  reason: string;
}

export interface RetrievedSource {
  source: string | null;
  citation: string | null;
  section: string;
  score: number | null;
}

export interface ExternalSource {
  id: string | null;
  title: string | null;
  publisher: string | null;
  citation: string | null;
  available: boolean;
  fetched_at: string | null;
  excerpt: string | null;
  reason: string | null;
}

export interface Meta {
  mode: string;
  saved_example: boolean;
  example_id: string | null;
  generated_at: string;
  how_made: string;
  report_generated: boolean;
  chat_model: string | null;
  embedding_model: string | null;
  corpus: Record<string, unknown> | null;
  data_current_through: Record<string, string> | null;
  horizon_manifest_source: string | null;
  data_source: string | null;
  banner: string | null;
  details: Record<string, unknown> | null;
}

export interface ReportResponse {
  request: string;
  report: string;
  grounding: Grounding;
  horizon_confidence: HorizonConfidence[];
  missing_data: MissingData[];
  tools_called: string[];
  tool_outputs: Record<string, any>;
  retrieved_sources: RetrievedSource[];
  external_sources: ExternalSource[];
  warnings: string[];
  quota: Record<string, unknown>;
  meta: Meta | null;
  slm?: SlmPassageCheck | null;
}

export interface SlmPassageCheck {
  enabled: boolean;
  model: string | null;
  skipped_reason: string | null;
  note: string;
  answers: {
    source: string | null;
    citation: string | null;
    answer: string | null;
    refused: boolean | null;
    error?: string;
    latency_ms?: number;
  }[];
}

export interface ExampleItem {
  id: string;
  label: string;
  shows: string;
  saved?: boolean;
  kind?: string;
  has_report?: boolean;
  body: ReportRequestBody;
}

export interface ReportRequestBody {
  request: string;
  region?: string | null;
  risk_types?: string[] | null;
  month?: string | null;
  crop?: string | null;
}

export interface ExamplesResponse {
  mode: string;
  examples: ExampleItem[];
  cost_warning: string;
  supported_regions: string[];
  supported_crops: string[];
}

export interface Health {
  status: string;
  mode: string;
  api_key_present: boolean;
  horizon_manifest_source: string;
  note: string;
}

export interface SlmStatus {
  configured: boolean;
  available: boolean;
  model: string | null;
  message: string;
}

export interface SlmAnswer {
  answer: string;
  refused: boolean;
  model: string | null;
  /** only on /slm/ask: where the context came from */
  sources?: { kind: string; name: string }[];
  truncated?: boolean;
}

export interface DemoRefusal {
  code: string;
  message: string;
  examples: string[];
}
