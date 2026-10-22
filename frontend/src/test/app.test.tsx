import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/client";
import App from "../App";
import HorizonBadges from "../components/HorizonBadges";
import Markdown from "../components/Markdown";
import ReportView from "../components/ReportView";
import { demoHealth, examples, liveHealth, report } from "./fixtures";

type Handler = (url: string, init?: RequestInit) => { status?: number; body: unknown };

function mockApi(handler: Handler) {
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    const { status = 200, body } = handler(String(url), init);
    return { ok: status < 400, status, json: async () => body } as Response;
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

afterEach(() => vi.unstubAllGlobals());

describe("home page", () => {
  it("lists the saved examples and explains that free text is off in demo mode", async () => {
    mockApi((url) => ({ body: url.endsWith("/health") ? demoHealth : examples }));
    renderAt("/analyst");
    expect(await screen.findByText("Drought next month, Rajasthan")).toBeInTheDocument();
    expect(screen.getByText("The number checker catching an invented figure")).toBeInTheDocument();
    expect(await screen.findByTestId("demo-explanation")).toHaveTextContent(/only the saved examples/i);
    expect(screen.getByRole("textbox")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Ask" })).toBeDisabled();
  });

  it("enables the text box in live mode and shows the backend's 409 message as it is", async () => {
    const message = "Demo mode answers only the saved examples. To ask your own question, run in live mode.";
    mockApi((url, init) => {
      if (url.endsWith("/health")) return { body: liveHealth };
      if (url.endsWith("/examples")) return { body: { ...examples, mode: "live" } };
      if (init?.method === "POST")
        return { status: 409, body: { detail: { code: "demo_free_text_disabled", message, examples: ["drought_one_month"] } } };
      return { status: 404, body: {} };
    });
    renderAt("/analyst");
    const box = await screen.findByRole("textbox");
    await waitFor(() => expect(box).toBeEnabled());
    await userEvent.type(box, "will it rain?");
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));
    const panel = await screen.findByTestId("refusal");
    expect(panel).toHaveTextContent(message);
    expect(panel).toHaveTextContent("drought_one_month");
  });

  it("live mode: a free-text question is posted to /report and the written report is shown", async () => {
    const written = report({
      request: "will it be dry in Barmer?",
      report: "## Summary\n\nThe skill is **0.2053** at t+1.",
      grounding: {
        status: "clean", grounded: true, total_checked: 1, unverified_numbers: [],
        report_missing: false, explanation: "All 1 numeric claims in this report were traced.",
      },
      meta: { ...report().meta!, mode: "live", saved_example: false, how_made: "live_generation",
        report_generated: true, banner: "Written live by the pipeline." },
    });
    const fetchMock = mockApi((url, init) => {
      if (url.endsWith("/health")) return { body: liveHealth };
      if (url.endsWith("/examples")) return { body: { ...examples, mode: "live" } };
      if (url.endsWith("/report") && init?.method === "POST") return { body: written };
      return { status: 404, body: {} };
    });
    renderAt("/analyst");
    const box = await screen.findByRole("textbox");
    await waitFor(() => expect(box).toBeEnabled());
    await userEvent.type(box, "will it be dry in Barmer?");
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));
    expect(await screen.findByRole("heading", { name: "Summary" })).toBeInTheDocument();
    expect(screen.getByTestId("grounding-status")).toHaveTextContent("clean");
    const post = fetchMock.mock.calls.find(([u]) => String(u).endsWith("/report"))!;
    expect(JSON.parse(String((post[1] as RequestInit).body))).toEqual({ request: "will it be dry in Barmer?" });
  });

  it("live mode: an error message from the backend (for example a missing key) is shown, not a broken page", async () => {
    const message = "Live mode needs a Gemini key in backend/.env (GEMINI_API_KEY). See the README.";
    mockApi((url, init) => {
      if (url.endsWith("/health")) return { body: liveHealth };
      if (url.endsWith("/examples")) return { body: { ...examples, mode: "live" } };
      if (init?.method === "POST") return { status: 503, body: { detail: message } };
      return { status: 404, body: {} };
    });
    renderAt("/analyst");
    const box = await screen.findByRole("textbox");
    await waitFor(() => expect(box).toBeEnabled());
    await userEvent.type(box, "will it rain?");
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(message);
    expect(screen.getByRole("button", { name: "Ask" })).toBeEnabled();
    expect(screen.getByText("Saved examples")).toBeInTheDocument();
  });

  it("says plainly when the backend cannot be reached", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("network"); }));
    renderAt("/analyst");
    expect(await screen.findByText(/backend not reachable/i)).toBeInTheDocument();
    expect(await screen.findByRole("alert")).toHaveTextContent(/Could not reach the backend/);
  });
});

describe("report page", () => {
  beforeEach(() => {
    mockApi((url) => ({ body: url.includes("/examples/") ? report() : demoHealth }));
  });

  it("always shows the banner, and says when the written report was not generated", async () => {
    renderAt("/analyst/report/drought_one_month");
    expect(await screen.findByTestId("banner")).toHaveTextContent(/Saved example/);
    expect(screen.getByText(/Not generated in this saved example/)).toBeInTheDocument();
    expect(screen.getByTestId("grounding-status")).toHaveTextContent("not_generated");
  });

  it("shows an error when an example does not exist", async () => {
    mockApi(() => ({ status: 404, body: { detail: "No saved example 'zzz'." } }));
    renderAt("/analyst/report/zzz");
    expect(await screen.findByRole("alert")).toHaveTextContent("No saved example 'zzz'.");
  });
});

describe("report view", () => {
  it("shows the backend's label, never one worked out from the score", () => {
    render(<HorizonBadges horizons={report().horizon_confidence} months={{ "t+3": "2025-03-01" }} />);
    expect(screen.getByText("Validated")).toBeInTheDocument();
    expect(screen.getByText("Weak / Directional")).toBeInTheDocument();
    // t+3 has a high score but the backend's label is "no skill": the page follows the label
    expect(screen.getByText("No skill")).toBeInTheDocument();
    expect(screen.getByText("0.9")).toBeInTheDocument();
    expect(screen.getByText(/t\+3 · 2025-03-01/)).toBeInTheDocument();
  });

  it("shows an unknown label as sent", () => {
    render(
      <HorizonBadges
        horizons={[{ horizon: "t+1", skill_score: 0.1, method: "x", label: "something new", reliable: false }]}
      />,
    );
    expect(screen.getAllByText("something new").length).toBeGreaterThan(0);
  });

  it("renders a written report and a warning with its flagged numbers", () => {
    const r = report({
      report: "> **UNVERIFIED FIGURES WARNING.** x\n\n## Summary\n\nThe skill is **0.2053**.\n\n* one\n* two",
      grounding: {
        status: "warning", grounded: false, total_checked: 3, unverified_numbers: ["0.42"],
        report_missing: false, explanation: "1 of 3 numeric claims could not be traced.",
      },
    });
    render(<ReportView report={r} />);
    expect(screen.getByTestId("grounding-status")).toHaveTextContent("warning");
    expect(screen.getByText("0.42")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Summary" })).toBeInTheDocument();
    expect(screen.getAllByText("0.2053").some((el) => el.tagName === "STRONG")).toBe(true);
    expect(screen.getAllByRole("listitem").length).toBeGreaterThan(1);
  });

  it("lists what the system does not have and the outside sources", () => {
    const r = report({
      external_sources: [
        { id: "imd", title: "IMD outlook", publisher: "IMD", citation: null, available: false,
          fetched_at: null, excerpt: null, reason: "timed out" },
      ],
    });
    render(<ReportView report={r} />);
    expect(screen.getByText(/reliable drought forecast at t\+3/)).toBeInTheDocument();
    expect(screen.getByText(/unavailable: timed out/)).toBeInTheDocument();
  });

  it("shows heat and crop results exactly as sent, including 'none' for a missing yield figure", () => {
    const r = report({
      tool_outputs: {
        forecast_heat_stress_risk: { month: "2024-05", heatwave_days: 7, severe_heatwave_days: 3,
          max_tmax_c: 48.0, forecast_available: false, note: "no usable skill" },
        assess_crop_impact: { crop: "bajra", month: "2025-09", dominant_risk: "none dominant",
          risk_reasoning: "because", yield_impact_pct: null,
          yield_impact_status: "no sourced yield-impact estimate available", confidence_label: "x" },
      },
    });
    render(<ReportView report={r} />);
    expect(screen.getByText(/false — no usable skill/)).toBeInTheDocument();
    expect(screen.getByText("no sourced yield-impact estimate available")).toBeInTheDocument();
    expect(screen.getByText("none")).toBeInTheDocument();
  });

  it("shows the passage check only when the step ran, with the 'not measured' statement", () => {
    const { rerender } = render(<ReportView report={report()} />);
    expect(screen.queryByTestId("slm-section")).toBeNull();
    rerender(
      <ReportView
        report={report({
          slm: {
            enabled: true, model: "fake-slm", skipped_reason: null, note: "Answers of a small separate model.",
            answers: [
              { source: "Doc A", citation: "https://example.org/a", answer: "40 C in the plains.", refused: false },
              { source: "Doc B", citation: null, answer: null, refused: true },
              { source: "Doc C", citation: null, answer: null, refused: null, error: "this passage is too long for the SLM" },
            ],
          },
        })}
      />,
    );
    const section = screen.getByTestId("slm-section");
    expect(section).toHaveTextContent(/effect on the quality of the report has not been measured/);
    expect(section).toHaveTextContent("40 C in the plains.");
    expect(section).toHaveTextContent("cannot answer the question");
    expect(section).toHaveTextContent("too long for the SLM");
  });

  it("copes with an answer that has no meta", () => {
    render(<ReportView report={report({ meta: null })} />);
    expect(screen.getByTestId("banner")).toHaveTextContent(/no banner/i);
  });
});

describe("markdown", () => {
  it("never turns text into HTML", () => {
    const { container } = render(<Markdown text={"<img src=x onerror=alert(1)> **bold**"} />);
    expect(container.querySelector("img")).toBeNull();
    expect(container).toHaveTextContent("<img src=x onerror=alert(1)>");
  });
});

describe("api errors", () => {
  it("recognises the demo refusal", () => {
    const e = new ApiError(409, "m", { code: "c", message: "m", examples: ["a"] });
    expect(e.demoRefusal).toEqual({ code: "c", message: "m", examples: ["a"] });
    expect(new ApiError(500, "m", "x").demoRefusal).toBeNull();
  });
});

describe("static pages", () => {
  it("Evidence names the corpus for every Track A number and the default search", async () => {
    mockApi(() => ({ body: demoHealth }));
    renderAt("/evidence");
    expect(await screen.findByRole("tab", { name: "Drought forecasting" })).toHaveAttribute("aria-selected", "true");
    await userEvent.click(screen.getByRole("tab", { name: "Retrieval (Track A)" }));
    expect(screen.getAllByText(/measured on the original 224-chunk corpus/).length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText(/Plain dense search stays the default/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("tab", { name: "Grounding checker" }));
    expect(screen.getByText(/Known defect/)).toBeInTheDocument();
  });

  it("SLM page: the service is optional, one adapter is published, the others are not", async () => {
    mockApi(() => ({ body: demoHealth }));
    renderAt("/slm");
    expect(await screen.findByTestId("slm-status")).toHaveTextContent(
      "The SLM service is optional and is not started by default",
    );
    expect(screen.getByTestId("slm-status")).toHaveTextContent(/does not claim|Nothing on this page claims/);
    expect(screen.getAllByRole("link", { name: "GHOSTBHAIYA/climate-slm-1.5b-sft" }).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/Only the 1.5B SFT adapter is published/)).toBeInTheDocument();
    expect(screen.getByText(/not a general chatbot/i)).toBeInTheDocument();
  });

  it("SLM Try it: off with the start-up steps when the service is not available", async () => {
    mockApi((url) => ({
      body: url.endsWith("/slm/status")
        ? { configured: false, available: false, model: null, message: "SLM service is not running; start it and set SLM_URL" }
        : demoHealth,
    }));
    renderAt("/slm");
    const note = await screen.findByTestId("slm-unavailable");
    expect(note).toHaveTextContent("PHASE5_PORT=8001");
    expect(note).toHaveTextContent("SLM_URL=http://127.0.0.1:8001");
    expect(screen.getByRole("button", { name: "Ask" })).toBeDisabled();
    expect(screen.getByLabelText(/^Question/)).toBeDisabled();
  });

  it("SLM Try it: sends the passage and question, shows the answer, refused flag and model", async () => {
    const fetchMock = mockApi((url, init) => {
      if (url.endsWith("/slm/status")) return { body: { configured: true, available: true, model: "fake-slm", message: "" } };
      if (url.endsWith("/slm/answer") && init?.method === "POST")
        return { body: { answer: "Rajasthan.", refused: false, model: "fake-slm" } };
      return { body: demoHealth };
    });
    renderAt("/slm");
    await waitFor(() => expect(screen.getByLabelText(/^Question/)).toBeEnabled());
    await userEvent.click(screen.getByLabelText("A passage I paste"));
    expect((screen.getByLabelText(/^Passage/) as HTMLTextAreaElement).value).toContain("Barmer");
    await userEvent.type(screen.getByLabelText(/^Question/), "In which state is Barmer?");
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));
    const result = await screen.findByTestId("slm-result");
    expect(result).toHaveTextContent("Rajasthan.");
    expect(result).toHaveTextContent("refused: false");
    expect(result).toHaveTextContent("model: fake-slm");
    const post = fetchMock.mock.calls.find(([u]) => String(u).endsWith("/slm/answer"))!;
    const sent = JSON.parse(String((post[1] as RequestInit).body));
    expect(Object.keys(sent).sort()).toEqual(["context", "question"]);
  });

  it("SLM Try it: shows a refusal and the backend's error message", async () => {
    let n = 0;
    mockApi((url, init) => {
      if (url.endsWith("/slm/status")) return { body: { configured: true, available: true, model: "fake-slm", message: "" } };
      if (url.endsWith("/slm/answer") && init?.method === "POST") {
        n += 1;
        return n === 1
          ? { body: { answer: "I cannot answer this from the provided context.", refused: true, model: "fake-slm" } }
          : { status: 503, body: { detail: "SLM service is not running; start it and set SLM_URL" } };
      }
      return { body: demoHealth };
    });
    renderAt("/slm");
    await waitFor(() => expect(screen.getByLabelText(/^Question/)).toBeEnabled());
    await userEvent.click(screen.getByLabelText("A passage I paste"));
    await userEvent.type(screen.getByLabelText(/^Question/), "A first question?");
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));
    expect(await screen.findByTestId("slm-result")).toHaveTextContent("refused: true");
    await userEvent.type(screen.getByLabelText(/^Question/), "And a second question?");
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("SLM service is not running");
  });

  it("SLM Try it: an over-long passage disables Ask", async () => {
    mockApi((url) => ({
      body: url.endsWith("/slm/status") ? { configured: true, available: true, model: "fake-slm", message: "" } : demoHealth,
    }));
    renderAt("/slm");
    await waitFor(() => expect(screen.getByLabelText(/^Question/)).toBeEnabled());
    await userEvent.click(screen.getByLabelText("A passage I paste"));
    await userEvent.type(screen.getByLabelText(/^Question/), "q?");
    const button = screen.getByRole("button", { name: "Ask" });
    await waitFor(() => expect(button).toBeEnabled());
    const box = screen.getByLabelText(/^Passage/);
    fireEvent.change(box, { target: { value: "x".repeat(4001) } });
    expect(button).toBeDisabled();
    expect(screen.getByRole("alert")).toHaveTextContent("4000");
  });

  it("Evidence SLM tab: the quoted tables say which adapter is published", async () => {
    mockApi(() => ({ body: demoHealth }));
    renderAt("/evidence");
    await userEvent.click(await screen.findByRole("tab", { name: "SLM" }));
    expect(screen.getAllByText("adapter not in this release").length).toBeGreaterThan(0);
    expect(screen.getAllByText("adapter published").length).toBeGreaterThanOrEqual(1);
  });

  it("About: limits, licence, repository and adapter links", async () => {
    mockApi(() => ({ body: demoHealth }));
    renderAt("/about");
    expect(await screen.findByRole("heading", { name: "Limits" })).toBeInTheDocument();
    expect(screen.getByText(/chunks\./)).toBeInTheDocument();
    expect(screen.getAllByText(/licensed under Apache-2.0/i).length).toBeGreaterThanOrEqual(1);
    const links = screen.getAllByRole("link").map((a) => a.getAttribute("href"));
    expect(links).toContain("https://github.com/ACUTE02/grounded-climate-analyst");
    expect(links).toContain("https://huggingface.co/GHOSTBHAIYA/climate-slm-1.5b-sft");
  });
});

describe("landing and navigation", () => {
  it("landing: what it is, two products, how it works, proof points with sources, demo wording, footer", async () => {
    mockApi(() => ({ body: demoHealth }));
    renderAt("/");
    expect(await screen.findByRole("heading", { level: 1 })).toHaveTextContent(/how far to trust each number/);
    expect(screen.getByText(/no key and no network/)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Forecasts with an honest label on every month" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "A small model that answers only from the text it is given" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "How it works" })).toBeInTheDocument();
    expect(screen.getAllByRole("heading", { level: 3 }).length).toBeGreaterThanOrEqual(5);
    // every proof number carries its source file in the hover title
    const proofs = screen.getByRole("heading", { name: "What was measured" }).parentElement!;
    const sources = proofs.querySelectorAll("span[title]");
    expect(sources.length).toBeGreaterThanOrEqual(4);
    sources.forEach((s) => expect(s.getAttribute("title")).toMatch(/\.json/));
    expect(screen.getByRole("contentinfo")).toHaveTextContent(/Apache-2\.0/);
    expect(screen.getByRole("contentinfo")).toHaveTextContent(/Open-Meteo/);
  });

  it("the top navigation links to the five pages and marks the current one", async () => {
    mockApi(() => ({ body: demoHealth }));
    renderAt("/evidence");
    const nav = await screen.findByRole("navigation", { name: "Main" });
    const names = Array.from(nav.querySelectorAll("a")).map((a) => a.textContent);
    expect(names).toEqual(["Home", "Analyst", "SLM", "Evidence", "About"]);
    expect(nav.querySelector('a[aria-current="page"]')).toHaveTextContent("Evidence");
  });

  it("the landing page product cards lead to /analyst and /slm", async () => {
    mockApi(() => ({ body: demoHealth }));
    renderAt("/");
    const hrefs = (await screen.findAllByRole("link")).map((a) => a.getAttribute("href"));
    expect(hrefs).toContain("/analyst");
    expect(hrefs).toContain("/slm");
  });

  it("an unknown address shows the landing page", async () => {
    mockApi(() => ({ body: demoHealth }));
    renderAt("/nothing-here");
    expect(await screen.findByRole("heading", { name: "How it works" })).toBeInTheDocument();
  });
});

describe("SLM chat", () => {
  const up = { configured: true, available: true, model: "fake-slm", message: "" };

  it("shows each question and answer as a turn, marks a refusal, and shows status, model and sources", async () => {
    let n = 0;
    mockApi((url, init) => {
      if (url.endsWith("/slm/status")) return { body: up };
      if (url.endsWith("/examples")) return { body: examples };
      if (url.endsWith("/slm/ask") && init?.method === "POST") {
        n += 1;
        return { body: n === 1
          ? { answer: "The next month is validated.", refused: false, model: "fake-slm",
              sources: [{ kind: "report", name: "report: q1" }, { kind: "corpus", name: "SPI guide" }], truncated: true }
          : { answer: "I cannot answer this from the provided context.", refused: true, model: "fake-slm",
              sources: [{ kind: "corpus", name: "SPI guide" }], truncated: false } };
      }
      return { body: demoHealth };
    });
    renderAt("/slm");
    await waitFor(() => expect(screen.getByTestId("slm-state")).toHaveTextContent("service running"));
    expect(screen.getByTestId("slm-model-badge")).toHaveTextContent("fake-slm");
    expect(screen.getByLabelText("The project's report and source corpus")).toBeChecked();
    expect(await screen.findByRole("option", { name: "Drought next month, Rajasthan" })).toBeInTheDocument();
    await userEvent.selectOptions(screen.getByLabelText("Report to include"), "drought_one_month");
    await userEvent.type(screen.getByLabelText(/^Question/), "Can I trust next month?");
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));
    expect(await screen.findByText("The next month is validated.")).toBeInTheDocument();
    expect(screen.getByTestId("slm-sources")).toHaveTextContent("report: report: q1; corpus: SPI guide (some text was cut to fit)");
    await userEvent.type(screen.getByLabelText(/^Question/), "What is the capital of France?");
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));
    const results = await screen.findAllByTestId("slm-result");
    expect(results).toHaveLength(2);
    expect(results[1]).toHaveTextContent("I cannot answer this from the provided context.");
    expect(results[1]).toHaveClass("refusal");
    expect(results[1]).toHaveTextContent("refused: true");
    expect(screen.getByText("What is the capital of France?")).toHaveClass("bubble", "q");
  });

  it("project mode posts only the question and the chosen report id to /slm/ask", async () => {
    const fetchMock = mockApi((url, init) => {
      if (url.endsWith("/slm/status")) return { body: up };
      if (url.endsWith("/examples")) return { body: examples };
      if (url.endsWith("/slm/ask") && init?.method === "POST")
        return { body: { answer: "ok", refused: false, model: "fake-slm", sources: [], truncated: false } };
      return { body: demoHealth };
    });
    renderAt("/slm");
    await screen.findByRole("option", { name: "Drought next month, Rajasthan" });
    await waitFor(() => expect(screen.getByLabelText(/^Question/)).toBeEnabled());
    await userEvent.type(screen.getByLabelText(/^Question/), "No report chosen?");
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));
    await screen.findByTestId("slm-result");
    const post = fetchMock.mock.calls.find(([u]) => String(u).endsWith("/slm/ask"))!;
    expect(JSON.parse(String((post[1] as RequestInit).body))).toEqual({ question: "No report chosen?" });
  });

  it("each turn sends the passage as currently edited", async () => {
    const fetchMock = mockApi((url, init) => {
      if (url.endsWith("/slm/status")) return { body: up };
      if (url.endsWith("/slm/answer") && init?.method === "POST") return { body: { answer: "ok", refused: false, model: "fake-slm" } };
      return { body: demoHealth };
    });
    renderAt("/slm");
    await waitFor(() => expect(screen.getByLabelText(/^Question/)).toBeEnabled());
    await userEvent.click(screen.getByLabelText("A passage I paste"));
    const passage = screen.getByLabelText(/^Passage/);
    await userEvent.type(screen.getByLabelText(/^Question/), "q?");
    fireEvent.change(passage, { target: { value: "My own passage." } });
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));
    await screen.findByTestId("slm-result");
    const post = fetchMock.mock.calls.find(([u]) => String(u).endsWith("/slm/answer"))!;
    expect(JSON.parse(String((post[1] as RequestInit).body)).context).toBe("My own passage.");
  });

  it("when the service is down the chat is off, the status says so, and the steps are shown", async () => {
    mockApi((url) => ({
      body: url.endsWith("/slm/status") ? { configured: false, available: false, model: null, message: "" } : demoHealth,
    }));
    renderAt("/slm");
    expect(await screen.findByTestId("slm-state")).toHaveTextContent("service not running");
    expect(screen.queryByTestId("slm-model-badge")).toBeNull();
    expect(screen.getByLabelText(/^Question/)).toBeDisabled();
    expect(screen.getByRole("button", { name: "Ask" })).toBeDisabled();
    expect(screen.getByTestId("slm-unavailable")).toHaveTextContent("PHASE5_PORT=8001");
  });
});
