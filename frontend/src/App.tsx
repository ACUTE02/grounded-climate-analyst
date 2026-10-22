import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import About from "./pages/About";
import Analyst from "./pages/Analyst";
import Evidence from "./pages/Evidence";
import Landing from "./pages/Landing";
import Report from "./pages/Report";
import Slm from "./pages/Slm";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Landing />} />
        <Route path="analyst" element={<Analyst />} />
        <Route path="analyst/report/:id" element={<Report />} />
        <Route path="slm" element={<Slm />} />
        <Route path="evidence" element={<Evidence />} />
        <Route path="about" element={<About />} />
        <Route path="*" element={<Landing />} />
      </Route>
    </Routes>
  );
}
