import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import ReportView from "../components/ReportView";
import { SkillChart } from "../components/SkillChart";
import droughtJson from "../data/drought.json";
import type { Drought } from "../data/types";
import { demoHealth, examples, report } from "./fixtures";

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

const health = {
  ...demoHealth,
  data_currency: { rajasthan: { data_current_through: "2024-12-01" }, barmer: { data_current_through: "2024-12-01" } },
  quota: { daily_call_budget: 20, calls_remaining_today: 20 },
};

afterEach(() => vi.unstubAllGlobals());

describe("landing preview (a real saved answer, not a mock-up)", () => {
  it("shows the saved example's question, its three labelled horizons and its number-check verdict", async () => {
    mockApi((url) => ({
      body: url.endsWith("/examples/drought_one_month")
        ? report({ grounding: { status: "clean", grounded: true, total_checked: 3, unverified_numbers: [], report_missing: false, explanation: "ok" } })
        : health,
    }));
    renderAt("/");
    const preview = await screen.findByTestId("landing-preview");
    await waitFor(() => expect(within(preview).getByText("What is the drought risk for Barmer?")).toBeInTheDocument());
    expect(within(preview).getByText("Validated")).toBeInTheDocument();
    expect(within(preview).getByText("Weak / Directional")).toBeInTheDocument();
    expect(within(preview).getByText("No skill")).toBeInTheDocument();
    expect(within(preview).getByText(/number check: clean/)).toBeInTheDocument();
    expect(within(preview).getByRole("link", { name: "Open the full report" })).toHaveAttribute("href", "/analyst/report/drought_one_month");
  });

  it("says plainly that the backend is needed when it cannot be reached, and the rest of the page still works", async () => {
    mockApi((url) => (url.includes("/examples/") ? { status: 500, body: { detail: "down" } } : { body: health }));
    renderAt("/");
    expect(await screen.findByText(/backend is not reachable/)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "How it works" })).toBeInTheDocument();
  });

  it("does not crash on a malformed answer", async () => {
    mockApi((url) => ({ body: url.includes("/examples/") ? { unexpected: true } : health }));
    renderAt("/");
    expect(await screen.findByText(/backend is not reachable/)).toBeInTheDocument();
    expect(screen.queryByTestId("page-error")).toBeNull();
  });
});

describe("skill chart", () => {
  const drought = droughtJson as unknown as Drought;

  it("draws one bar per horizon, coloured by the label, with an accessible description and a table fallback", () => {
    const rows = drought.regions["rajasthan"];
    const { container } = render(<SkillChart region="rajasthan" rows={rows} />);
    const svg = screen.getByRole("img");
    expect(svg).toHaveAttribute("aria-label", expect.stringContaining("t+1 Validated"));
    expect(svg.getAttribute("aria-label")).toContain("No skill");
    expect(container.querySelectorAll("rect.bar-validated")).toHaveLength(1);
    expect(container.querySelectorAll("rect.bar-noskill")).toHaveLength(1);
    expect(container.querySelectorAll("rect").length).toBe(rows.length);
    const table = screen.getByRole("table");
    expect(within(table).getAllByRole("row")).toHaveLength(rows.length + 1);
  });

  it("puts a negative skill below the zero line", () => {
    const rows = drought.regions["rajasthan"];
    const { container } = render(<SkillChart region="rajasthan" rows={rows} />);
    const axis = Number(container.querySelector("line.axis")!.getAttribute("y1"));
    const last = container.querySelector("rect.bar-noskill")!;
    expect(Number(last.getAttribute("y"))).toBeGreaterThanOrEqual(axis);
    const first = container.querySelector("rect.bar-validated")!;
    expect(Number(first.getAttribute("y")) + Number(first.getAttribute("height"))).toBeLessThanOrEqual(axis + 0.01);
  });
});

describe("analyst page states", () => {
  it("shows the backend's real status (mode, key, data date, quota, coverage)", async () => {
    mockApi((url) => ({ body: url.endsWith("/examples") ? { ...examples, quota: health.quota } : health }));
    renderAt("/analyst");
    const panel = await screen.findByTestId("status-panel");
    await waitFor(() => expect(panel).toHaveTextContent("2024-12-01"));
    expect(panel).toHaveTextContent("demo");
    expect(panel).toHaveTextContent("20 of 20 calls left");
    await waitFor(() => expect(panel).toHaveTextContent("barmer, rajasthan"));
  });

  it("shows an error when the examples cannot be loaded, and a working Try again button", async () => {
    let fail = true;
    mockApi((url) => {
      if (url.endsWith("/examples")) return fail ? { status: 500, body: { detail: "backend is warming up" } } : { body: examples };
      return { body: health };
    });
    renderAt("/analyst");
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("backend is warming up");
    fail = false;
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText("Drought next month, Rajasthan")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).toBeNull();
  });
});

describe("report page", () => {
  it("summarises trust, the number check and the data date, and offers a section menu", () => {
    render(<ReportView report={report()} />);
    const summary = screen.getByTestId("summary");
    expect(summary).toHaveTextContent("1 of 3 horizons validated");
    expect(summary).toHaveTextContent("not_generated");
    expect(summary).toHaveTextContent("2024-12-01");
    const toc = screen.getByRole("navigation", { name: "On this page" });
    const labels = within(toc).getAllByRole("button").map((b) => b.textContent);
    expect(labels).toEqual(["How far to trust it", "What the system lacks", "Number check", "Written report", "Sources and run details"]);
  });

  it("scrolls to a section when its menu entry is pressed", async () => {
    const scrollIntoView = vi.fn();
    Element.prototype.scrollIntoView = scrollIntoView;
    render(<ReportView report={report()} />);
    await userEvent.click(screen.getByRole("button", { name: "Number check" }));
    expect(scrollIntoView).toHaveBeenCalled();
    const target = scrollIntoView.mock.contexts[0] as HTMLElement;
    expect(target.id).toBe("h-ground");
  });

  it("keeps the run details inside the banner but folded away", () => {
    render(<ReportView report={report()} />);
    const details = screen.getByTestId("banner").querySelector("details")!;
    expect(details).not.toHaveAttribute("open");
    expect(details).toHaveTextContent("mode: demo");
  });
});

describe("shell", () => {
  it("has a skip link that moves focus to the main content", async () => {
    mockApi(() => ({ body: health }));
    renderAt("/about");
    const skip = await screen.findByRole("link", { name: "Skip to content" });
    await userEvent.click(skip);
    expect(document.getElementById("main")).toHaveFocus();
  });

  it("names its landmarks", async () => {
    mockApi(() => ({ body: health }));
    renderAt("/about");
    expect(await screen.findByRole("banner")).toBeInTheDocument();
    expect(screen.getByRole("main")).toBeInTheDocument();
    expect(screen.getByRole("contentinfo")).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Main" })).toBeInTheDocument();
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
  });
});
