import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import About from "./pages/About";
import Evidence from "./pages/Evidence";
import Home from "./pages/Home";
import Report from "./pages/Report";
import Slm from "./pages/Slm";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="report/:id" element={<Report />} />
        <Route path="evidence" element={<Evidence />} />
        <Route path="slm" element={<Slm />} />
        <Route path="about" element={<About />} />
        <Route path="*" element={<Home />} />
      </Route>
    </Routes>
  );
}
