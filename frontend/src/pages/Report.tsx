import { useEffect, useState } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import { ApiError, getExample } from "../api/client";
import type { ReportResponse } from "../api/types";
import ReportView from "../components/ReportView";

export default function Report() {
  const { id = "" } = useParams();
  const location = useLocation();
  const passed = (location.state as { report?: ReportResponse } | null)?.report;
  const [report, setReport] = useState<ReportResponse | null>(passed ?? null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (id === "live") return; // a live answer is handed over by the Ask page
    setReport(null);
    setError(null);
    getExample(id)
      .then(setReport)
      .catch((e: unknown) => setError(e instanceof ApiError ? e.message : String(e)));
  }, [id]);

  if (id === "live" && !report) {
    return (
      <div className="calm">
        This live answer is no longer in memory. <Link to="/analyst">Ask again</Link>.
      </div>
    );
  }
  if (error) {
    return (
      <div className="error" role="alert">
        {error} <Link to="/analyst">Back to the examples</Link>
      </div>
    );
  }
  if (!report)
    return (
      <div aria-busy="true">
        <p className="sr-only" role="status">Loading the saved example…</p>
        <div className="skeleton" style={{ minHeight: 56, marginBottom: 16 }} />
        <div className="skeleton" style={{ minHeight: 220 }} />
      </div>
    );
  return (
    <>
      <ReportView report={report} crumb={<Link to="/analyst">← Analyst</Link>} />
    </>
  );
}
