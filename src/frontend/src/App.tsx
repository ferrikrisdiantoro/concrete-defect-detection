import { Routes, Route, Link, NavLink } from "react-router-dom";
import Analyze from "./pages/Analyze";
import History from "./pages/History";
import Navbar from "./components/Navbar";

export default function App() {
  return (
    <div className="min-h-screen bg-bg antialiased">
      <Navbar />
      <main className="mx-auto w-full max-w-7xl px-4 md:px-8">
        <Routes>
          <Route path="/" element={<Analyze />} />
          <Route path="/history" element={<History />} />
          <Route path="*" element={
            <div className="py-10 text-sm">
              Halaman tidak ditemukan. <NavLink to="/" className="text-primary underline">Kembali</NavLink>
            </div>
          }/>
        </Routes>
      </main>
      <footer className="mx-auto w-full max-w-7xl border-t border-zinc-200 px-4 py-8 text-xs text-muted md:px-8">
        © {new Date().getFullYear()} Crack Detector • FastAPI + React
      </footer>
    </div>
  );
}
