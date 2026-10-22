import type {
  DemoRefusal,
  ExamplesResponse,
  Health,
  ReportRequestBody,
  ReportResponse,
  SlmAnswer,
  SlmStatus,
} from "./types";

export const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://127.0.0.1:8000";

/** Thrown for any answer that is not a success; `status` is the HTTP status (0 = no answer). */
export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, message: string, detail?: unknown) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
  /** The demo-mode refusal (HTTP 409) with the backend's own message, if that is what this is. */
  get demoRefusal(): DemoRefusal | null {
    const d = this.detail as Partial<DemoRefusal> | undefined;
    if (this.status === 409 && d && typeof d.message === "string") {
      return { code: d.code ?? "", message: d.message, examples: d.examples ?? [] };
    }
    return null;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new ApiError(0, `Could not reach the backend at ${API_BASE}. Is it running?`);
  }
  let body: any = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    const detail = body?.detail;
    const message =
      typeof detail === "string" ? detail : detail?.message ?? `HTTP ${response.status}`;
    throw new ApiError(response.status, message, detail);
  }
  return body as T;
}

export const getHealth = () => request<Health>("/health");
export const getExamples = () => request<ExamplesResponse>("/examples");
export const getExample = (id: string) =>
  request<ReportResponse>(`/examples/${encodeURIComponent(id)}`);
export const getSlmStatus = () => request<SlmStatus>("/slm/status");
export const postSlmAnswer = (context: string, question: string) =>
  request<SlmAnswer>("/slm/answer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ context, question }),
  });
export const postSlmAsk = (question: string, reportId: string | null) =>
  request<SlmAnswer>("/slm/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(reportId ? { question, report_id: reportId } : { question }),
  });
export const postReport = (body: ReportRequestBody) =>
  request<ReportResponse>("/report", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
