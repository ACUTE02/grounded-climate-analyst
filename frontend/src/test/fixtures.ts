import type { ExamplesResponse, Health, ReportResponse } from "../api/types";

export const demoHealth: Health = {
  status: "ok",
  mode: "demo",
  api_key_present: false,
  horizon_manifest_source: "generated",
  note: "test",
};
export const liveHealth: Health = { ...demoHealth, mode: "live" };

export const examples: ExamplesResponse = {
  mode: "demo",
  cost_warning: "Demo mode: saved examples only, no Gemini calls are made.",
  supported_regions: ["barmer", "rajasthan"],
  supported_crops: ["bajra", "wheat"],
  examples: [
    {
      id: "drought_one_month",
      label: "Drought next month, Rajasthan",
      shows: "The one validated horizon.",
      kind: "tools_only",
      has_report: false,
      body: { request: "q1" },
    },
    {
      id: "grounding_catch",
      label: "The number checker catching an invented figure",
      shows: "A flagged number.",
      kind: "constructed_fixture",
      has_report: true,
      body: { request: "q2" },
    },
  ],
};

// A deliberately odd pair: a high score with a "no skill" label. The page must show the
// label it was sent, not one it works out from the score.
export function report(over: Partial<ReportResponse> = {}): ReportResponse {
  return {
    request: "What is the drought risk for Barmer?",
    report: "",
    grounding: {
      status: "not_generated",
      grounded: false,
      total_checked: 0,
      unverified_numbers: [],
      report_missing: true,
      explanation: "No report was produced, so there was nothing to verify.",
    },
    horizon_confidence: [
      { horizon: "t+1", skill_score: 0.2053, method: "direct", label: "validated", reliable: true },
      { horizon: "t+2", skill_score: 0.0438, method: "direct", label: "weak/directional", reliable: false },
      {
        horizon: "t+3",
        skill_score: 0.9,
        method: "direct",
        label: "no skill — shown for context only, do not rely on this figure",
        reliable: false,
      },
    ],
    missing_data: [
      { what: "reliable drought forecast at t+3", available: false, reason: "no skill — shown for context only" },
    ],
    tools_called: ["forecast_drought_risk"],
    tool_outputs: {
      forecast_drought_risk: {
        horizon_confidence: [
          { horizon: "t+1", month: "2025-01-01" },
          { horizon: "t+2", month: "2025-02-01" },
          { horizon: "t+3", month: "2025-03-01" },
        ],
        forecast_anchor_month: "2024-12-01",
        predicted_values: [0.2, 0.1, 0.0],
      },
    },
    retrieved_sources: [],
    external_sources: [],
    warnings: [],
    quota: {},
    meta: {
      mode: "demo",
      saved_example: true,
      example_id: "x",
      generated_at: "2026-10-22T10:00:00+00:00",
      how_made: "tools_only",
      report_generated: false,
      chat_model: null,
      embedding_model: null,
      corpus: null,
      data_current_through: { barmer: "2024-12-01" },
      horizon_manifest_source: "generated",
      data_source: "snapshot",
      banner: "Saved example. These are the real outputs of the forecasting tools.",
      details: null,
    },
    ...over,
  };
}
