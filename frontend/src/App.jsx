import { Navigate, Route, Routes } from "react-router-dom";
import AppShell from "./components/layout/AppShell";
import Dashboard from "./pages/Dashboard";
import Inference from "./pages/Inference";
import JobHistory from "./pages/JobHistory";
import Models from "./pages/Models";
import Workers from "./pages/Workers";

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Dashboard />} />
        <Route path="workers" element={<Workers />} />
        <Route path="models" element={<Models />} />
        <Route path="inference" element={<Inference />} />
        <Route path="jobs" element={<JobHistory />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
