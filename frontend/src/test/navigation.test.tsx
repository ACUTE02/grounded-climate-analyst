import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HashRouter, MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import ErrorBoundary from "../components/ErrorBoundary";
import Layout from "../components/Layout";
import { demoHealth, examples } from "./fixtures";

// Regression: clicking from one page to another (no reload) blanked the site with "l is not a function",
// because an effect returned the value of window.scrollTo. A real browser may return a value there; jsdom does not.
// Here scrollTo returns a truthy value, and ANY console.error (React reports render loops, bad effects and
// uncaught errors there) fails the test.

const PAGES: [string, RegExp][] = [
  ["Analyst", /Ask about drought, heat and crops/],
  ["SLM", /answers from the passage you give it/],
  ["Evidence", /What was measured, and where it was read from/],
  ["About", /What this release is, and what it is not/],
  ["Home", /how far to trust each number/],
];

let consoleError: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
  vi.stubGlobal("scrollTo", vi.fn(() => ({ not: "undefined" })));
  vi.stubGlobal("fetch", vi.fn(async (url: string) => ({
    ok: true, status: 200,
    json: async () => (String(url).endsWith("/examples") ? examples
      : String(url).endsWith("/slm/status") ? { configured: false, available: false, model: null, message: "" }
      : demoHealth),
  } as Response)));
});
afterEach(() => {
  vi.unstubAllGlobals();
  consoleError.mockRestore();
  window.location.hash = "";
});

async function clickThrough() {
  const nav = await screen.findByRole("navigation", { name: "Main" });
  for (const [name, heading] of PAGES) {
    await userEvent.click(Array.from(nav.querySelectorAll("a")).find((a) => a.textContent === name)!);
    expect(await screen.findByRole("heading", { level: 1 })).toHaveTextContent(heading);
    expect(screen.getByRole("navigation", { name: "Main" })).toBeInTheDocument();
    expect(screen.getByRole("contentinfo")).toBeInTheDocument();
  }
  expect(consoleError).not.toHaveBeenCalled();
}

describe("client-side navigation", () => {
  it("MemoryRouter: every nav link renders its page with no console error, even if scrollTo returns a value", async () => {
    render(<MemoryRouter initialEntries={["/"]}><App /></MemoryRouter>);
    await clickThrough();
  });

  it("HashRouter (as in the site): the same clicks render every page", async () => {
    window.location.hash = "#/";
    render(<HashRouter><App /></HashRouter>);
    await clickThrough();
    expect(window.location.hash).toBe("#/");
  });

  it("goes back and forth between two pages many times without a render loop", async () => {
    render(<MemoryRouter initialEntries={["/analyst"]}><App /></MemoryRouter>);
    const nav = await screen.findByRole("navigation", { name: "Main" });
    const link = (name: string) => Array.from(nav.querySelectorAll("a")).find((a) => a.textContent === name)!;
    for (let i = 0; i < 6; i++) {
      await userEvent.click(link("Evidence"));
      await screen.findByRole("tab", { name: "Drought forecasting" });
      await userEvent.click(link("Analyst"));
      await screen.findByRole("heading", { name: "Saved examples" });
    }
    expect(consoleError).not.toHaveBeenCalled();
  });

  it("scrolls to the top on a route change", async () => {
    const scrollTo = vi.fn();
    vi.stubGlobal("scrollTo", scrollTo);
    render(<MemoryRouter initialEntries={["/"]}><App /></MemoryRouter>);
    const nav = await screen.findByRole("navigation", { name: "Main" });
    scrollTo.mockClear();
    await userEvent.click(Array.from(nav.querySelectorAll("a")).find((a) => a.textContent === "About")!);
    await waitFor(() => expect(scrollTo).toHaveBeenCalledWith(0, 0));
  });
});

describe("error boundary", () => {
  function Boom(): never {
    throw new Error("boom in a page");
  }

  it("shows a message instead of a blank page and keeps the navigation and footer", async () => {
    render(
      <MemoryRouter>
        <Layout />
      </MemoryRouter>,
    );
    // the layout itself renders; now a throwing child inside a boundary
    const { container } = render(<ErrorBoundary><Boom /></ErrorBoundary>);
    expect(container.querySelector('[data-testid="page-error"]')).toHaveTextContent("This page could not be shown.");
    expect(container).toHaveTextContent("boom in a page");
    expect(screen.getByRole("navigation", { name: "Main" })).toBeInTheDocument();
    expect(screen.getByRole("contentinfo")).toBeInTheDocument();
  });

  it("a page that throws is contained, and navigating away shows the next page", async () => {
    function Wrapper({ throwing }: { throwing: boolean }) {
      return <ErrorBoundary key={String(throwing)}>{throwing ? <Boom /> : <p>fine page</p>}</ErrorBoundary>;
    }
    const { rerender } = render(<Wrapper throwing />);
    expect(screen.getByTestId("page-error")).toBeInTheDocument();
    await act(async () => rerender(<Wrapper throwing={false} />));
    expect(screen.getByText("fine page")).toBeInTheDocument();
    expect(screen.queryByTestId("page-error")).toBeNull();
  });
});
