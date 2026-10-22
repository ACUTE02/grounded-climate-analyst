import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const root = path.resolve(__dirname, "..", "..");
const read = (...p: string[]) => readFileSync(path.join(root, ...p), "utf-8");

describe("tab icon and title", () => {
  it("index.html links the SVG icon and has the repository-matching title and a description", () => {
    const html = read("index.html");
    expect(html).toContain('<link rel="icon" type="image/svg+xml" href="/favicon.svg" />');
    expect(html).toContain("<title>Grounded Climate Analyst</title>");
    expect(html).toMatch(/<meta name="description" content="[^"]{20,200}"/);
  });

  it("the icon is a small valid SVG that uses the site palette", () => {
    const svg = read("public", "favicon.svg");
    expect(svg.startsWith("<svg")).toBe(true);
    expect(svg).toContain('viewBox="0 0 64 64"');
    expect(svg.length).toBeLessThan(2000);
    expect(svg).not.toMatch(/<script|href=|<image|<text|<font/i);
    for (const colour of ["#1c2620", "#f3eee3"]) expect(svg).toContain(colour);
  });

  it("nginx answers /favicon.ico with a redirect and a missing asset with 404, not index.html", () => {
    const conf = read("nginx.conf");
    expect(conf).toMatch(/location = \/favicon\.ico \{\s*return 301 \/favicon\.svg;/);
    expect(conf).toMatch(/location \/assets\/ \{\s*try_files \$uri =404;/);
  });

  // The build output exists after `npm run build`; this checks it when it is there.
  it.skipIf(!existsSync(path.join(root, "dist", "index.html")))("dist has favicon.svg and the link", () => {
    expect(existsSync(path.join(root, "dist", "favicon.svg"))).toBe(true);
    expect(read("dist", "index.html")).toMatch(/href="\.?\/favicon\.svg"/);
  });
});
