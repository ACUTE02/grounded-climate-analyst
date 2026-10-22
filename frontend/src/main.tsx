import "@fontsource-variable/bricolage-grotesque";
import "@fontsource-variable/fraunces";
import "@fontsource-variable/martian-mono";
import "./theme.css";

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { HashRouter } from "react-router-dom";
import App from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <HashRouter>
      <App />
    </HashRouter>
  </StrictMode>,
);
