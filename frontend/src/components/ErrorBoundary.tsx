import { Component, type ErrorInfo, type ReactNode } from "react";

interface State {
  error: Error | null;
}

/** Keeps a failure on one page from blanking the whole site: the navigation and footer stay, and the page says what happened. */
export default class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Page failed to render:", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="error" role="alert" data-testid="page-error">
          <h2 style={{ marginTop: 0 }}>This page could not be shown.</h2>
          <p className="small" style={{ margin: 0 }}>
            {this.state.error.message || "Unknown error"}. Use the navigation above to go to another page, or reload.
          </p>
        </div>
      );
    }
    return this.props.children;
  }
}
