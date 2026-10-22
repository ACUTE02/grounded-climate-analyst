import { useOutletContext } from "react-router-dom";
import type { Health } from "../api/types";

export interface ShellContext {
  health: Health | null;
  offline: boolean;
}

export const useShell = () => useOutletContext<ShellContext>();
