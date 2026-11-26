import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { Menu, Moon, Network, Sun, X } from "lucide-react";
import NetworkBackground from "../common/NetworkBackground";

const navigation = [
  { label: "Overview", to: "/", end: true },
  { label: "Workers", to: "/workers" },
  { label: "Models", to: "/models" },
  { label: "Inference", to: "/inference" },
  { label: "Job History", to: "/jobs" },
];
const THEME_STORAGE_KEY = "dmg-theme-v2";

function Brand() {
  return (
    <div className="flex shrink-0 items-center gap-2.5">
      <span className="grid h-10 w-10 place-items-center rounded-xl bg-dmg-accent text-dmg-accent-text shadow-sm shadow-dmg-accent/20">
        <Network size={18} strokeWidth={1.8} />
      </span>
      <span className="leading-tight">
        <span className="block text-base font-bold tracking-[0.12em] text-dmg-accent">DMG</span>
        <span className="mt-1 hidden text-xs text-dmg-subtle sm:block">
          Distributed Memory Grid
        </span>
      </span>
    </div>
  );
}

function NavigationLinks({ mobile = false, onNavigate }) {
  return (
    <nav
      aria-label="Main navigation"
      className={mobile ? "grid gap-1.5" : "hidden h-full items-center gap-1 lg:flex"}
    >
      {navigation.map(({ label, to, end }) => (
        <NavLink
          key={to}
          to={to}
          end={end}
          onClick={onNavigate}
          className={({ isActive }) =>
            `text-base transition-colors ${
              mobile
                ? `rounded-lg px-3.5 py-3 ${
                    isActive
                      ? "bg-dmg-accent-soft font-semibold text-dmg-accent"
                      : "text-dmg-muted hover:bg-dmg-surface-alt hover:text-dmg-text"
                  }`
                : `flex h-10 items-center rounded-lg px-3 ${
                    isActive
                      ? "bg-dmg-accent-soft font-semibold text-dmg-accent"
                      : "text-dmg-muted hover:bg-dmg-surface-alt hover:text-dmg-text"
                  }`
            }`
          }
        >
          {label}
        </NavLink>
      ))}
    </nav>
  );
}

function ClusterStatus() {
  return (
    <div className="flex items-center gap-2.5 rounded-md border border-dmg-line/15 bg-dmg-surface px-3 py-2">
      <span className="h-2 w-2 rounded-full bg-dmg-indicator" aria-hidden="true" />
      <span className="text-sm font-medium text-dmg-muted">Not connected</span>
    </div>
  );
}

function ThemeToggle({ theme, onToggle }) {
  const isDark = theme === "dark";
  const label = isDark ? "Switch to light theme" : "Switch to dark theme";
  const Icon = isDark ? Sun : Moon;

  return (
    <button
      type="button"
      onClick={onToggle}
      aria-label={label}
      title={label}
      className="inline-flex h-10 items-center gap-2 rounded-lg border border-dmg-line/20 bg-dmg-surface px-3.5 text-sm font-medium text-dmg-muted transition-colors hover:bg-dmg-surface-alt hover:text-dmg-text"
    >
      <Icon size={15} strokeWidth={1.8} />
      <span className="hidden sm:inline">{isDark ? "Dark" : "Light"}</span>
    </button>
  );
}

export default function AppShell() {
  const { pathname } = useLocation();
  const isDashboard = pathname === "/";
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [theme, setTheme] = useState(() =>
    localStorage.getItem(THEME_STORAGE_KEY) === "light" ? "light" : "dark",
  );

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(THEME_STORAGE_KEY, theme);
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute("content", theme === "dark" ? "#0a0e0c" : "#fafbf9");
  }, [theme]);

  function toggleTheme() {
    setTheme((currentTheme) => (currentTheme === "light" ? "dark" : "light"));
  }

  return (
    <>
      <NetworkBackground />
      <div className="relative z-10 min-h-screen text-dmg-text">
        <header className="sticky top-0 z-30 border-b border-dmg-line/10 bg-dmg-surface/85 backdrop-blur">
          <div className="mx-auto flex h-[76px] max-w-[1440px] items-center justify-between gap-5 px-5 sm:px-8">
            <div className="flex h-full items-center gap-6 xl:gap-9">
              <Brand />
              <NavigationLinks />
            </div>
            <div className="flex shrink-0 items-center gap-2">
              <div className="hidden md:block">
                <ClusterStatus />
              </div>
              <ThemeToggle theme={theme} onToggle={toggleTheme} />
              <button
                type="button"
                aria-label={mobileMenuOpen ? "Close navigation" : "Open navigation"}
                aria-expanded={mobileMenuOpen}
                onClick={() => setMobileMenuOpen((open) => !open)}
                className="grid h-10 w-10 place-items-center rounded-lg border border-dmg-line/20 bg-dmg-surface text-dmg-muted lg:hidden"
              >
                {mobileMenuOpen ? <X size={17} /> : <Menu size={17} />}
              </button>
            </div>
          </div>
          {mobileMenuOpen && (
            <div className="border-t border-dmg-line/10 bg-dmg-surface/95 px-5 py-4 shadow-lg backdrop-blur sm:px-8 lg:hidden">
              <div className="mx-auto max-w-[1440px]">
                <NavigationLinks mobile onNavigate={() => setMobileMenuOpen(false)} />
                <div className="mt-3 md:hidden">
                  <ClusterStatus />
                </div>
              </div>
            </div>
          )}
        </header>

        <main
          className={
            isDashboard
              ? "w-full"
              : "mx-auto w-full max-w-[1280px] px-5 py-11 sm:px-8 sm:py-14"
          }
        >
          <Outlet />
        </main>
        <footer className="mx-auto max-w-[1280px] border-t border-dmg-line/15 px-5 py-6 text-sm text-dmg-subtle sm:px-8">
          DMG <span className="px-1.5">·</span> Distributed Memory Grid
        </footer>
      </div>
    </>
  );
}
