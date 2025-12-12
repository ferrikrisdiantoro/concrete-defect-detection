import { Link, NavLink } from "react-router-dom";

export default function Navbar() {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-zinc-200 bg-white/80 backdrop-blur supports-[backdrop-filter]:bg-white/60">
      <div className="mx-auto flex h-16 w-full max-w-7xl items-center justify-between px-4 md:px-8">
        <Link to="/" className="group inline-flex items-center gap-2">
          {/* Logo dari public/logo.png */}
          <img
            src="/logo.png"
            alt="Crack Detector"
            className="h-8 w-auto rounded-md"
          />
          {/* Teks disembunyikan untuk aksesibilitas; hapus jika ingin logo-only */}
          <span className="sr-only">Crack Detector</span>
        </Link>

        <nav className="flex items-center gap-1 sm:gap-2">
          <NavLink to="/" className="navlink">Analyze</NavLink>
          <NavLink to="/history" className="navlink">History</NavLink>
        </nav>
      </div>
    </header>
  );
}
