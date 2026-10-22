import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { getHealth } from "../api/client";
import type { Health } from "../api/types";

export default function Layout() {
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setOffline(true));
  }, []);

  return (
    <div className="shell">
      <header className="topbar">
        <NavLink to="/" className="brand">
          Climate Suite
        </NavLink>
        <nav className="nav" aria-label="Main">
          <NavLink to="/" end>
            Ask
          </NavLink>
          <NavLink to="/evidence">Evidence</NavLink>
          <NavLink to="/slm">SLM</NavLink>
          <NavLink to="/about">About this release</NavLink>
        </nav>
        <span className="spacer" />
        {health && (
          <span className="pill accent" title={health.note}>
            {health.mode} mode
          </span>
        )}
        {offline && <span className="pill warm">backend not reachable</span>}
      </header>
      <main>
        <Outlet context={{ health, offline }} />
      </main>
    </div>
  );
}
