import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { getHealth } from "../api/client";
import type { Health } from "../api/types";
import ErrorBoundary from "./ErrorBoundary";

export const REPO_URL = "https://github.com/ACUTE02/grounded-climate-analyst";
export const ADAPTER_URL = "https://huggingface.co/GHOSTBHAIYA/climate-slm-1.5b-sft";

/** A small mark: a sun over a horizon line, above three bars (the three forecast horizons). */
function BrandMark() {
  return (
    <svg width="22" height="22" viewBox="0 0 64 64" aria-hidden="true">
      <rect width="64" height="64" rx="12" fill="#151b26" />
      <path d="M15 34a17 17 0 0 1 34 0z" fill="#e0884f" />
      <rect x="8" y="34" width="48" height="3" fill="#f6f7f9" />
      <rect x="13" y="42" width="10" height="14" fill="#4aa388" />
      <rect x="27" y="46" width="10" height="10" fill="#e0ac3f" />
      <rect x="41" y="50" width="10" height="6" fill="#d05a55" />
    </svg>
  );
}

export default function Layout() {
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);
  const { pathname } = useLocation();

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setOffline(true));
  }, []);

  // A block body on purpose: an effect must return nothing or a cleanup function, never scrollTo's result.
  useEffect(() => {
    window.scrollTo?.(0, 0);
  }, [pathname]);

  return (
    <>
      <a className="skip-link" href="#main" onClick={(e) => { e.preventDefault(); document.getElementById("main")?.focus(); }}>
        Skip to content
      </a>
      <header className="topbar">
        <div className="wrap">
          <NavLink to="/" className="brand">
            <BrandMark />
            Climate Suite
          </NavLink>
          <nav className="nav" aria-label="Main">
            <NavLink to="/" end>Home</NavLink>
            <NavLink to="/analyst">Analyst</NavLink>
            <NavLink to="/slm">SLM</NavLink>
            <NavLink to="/evidence">Evidence</NavLink>
            <NavLink to="/about">About</NavLink>
          </nav>
          <span className="spacer" />
          {health && (
            <span className="pill accent" title={health.note}>
              <span className="dot" aria-hidden="true" />
              {health.mode} mode
            </span>
          )}
          {offline && <span className="pill warm">backend not reachable</span>}
          <a className="repo" href={REPO_URL}>Source</a>
        </div>
      </header>
      <main className="wrap" id="main" tabIndex={-1} style={{ outline: "none" }}>
        <ErrorBoundary key={pathname}>
          <Outlet context={{ health, offline }} />
        </ErrorBoundary>
      </main>
      <footer className="footer">
        <div className="wrap cols">
          <div>
            <h4>Climate Suite</h4>
            <p>
              Licensed under Apache-2.0. Please credit Ayushmaan Gupta and link back to{" "}
              <a href={REPO_URL}>the repository</a>.
            </p>
          </div>
          <div>
            <h4>Data</h4>
            <p>
              Weather data by <a href="https://open-meteo.com/">Open-Meteo.com</a> (CC BY 4.0), modified
              Copernicus Climate Change Service information; ENSO (ONI) and Indian Ocean Dipole (DMI) series from
              NOAA PSL; DMI computed from HadISST1.1 (Met Office, Open Government Licence). Details: NOTICE in
              the repository.
            </p>
          </div>
          <div>
            <h4>Models</h4>
            <p>
              <a href={ADAPTER_URL}>GHOSTBHAIYA/climate-slm-1.5b-sft</a> (Apache-2.0) on Qwen/Qwen2.5-1.5B-Instruct
              (Apache-2.0). Not stored in the repository.
            </p>
          </div>
        </div>
      </footer>
    </>
  );
}
