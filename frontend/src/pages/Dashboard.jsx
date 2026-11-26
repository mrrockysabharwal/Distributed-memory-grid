import { ArrowDownRight, ArrowRight, Boxes, Cpu, Waypoints } from "lucide-react";
import { Link } from "react-router-dom";

const statusItems = [
  { label: "Cluster", value: "Not connected", detail: "No backend connection", icon: Waypoints },
  { label: "Workers", value: "No data", detail: "No workers reported", icon: Cpu },
  { label: "Models", value: "No data", detail: "Model catalog unavailable", icon: Boxes },
];

export default function Dashboard() {
  return (
    <section className="hero-section relative flex min-h-[calc(100svh-76px)] w-full items-stretch overflow-hidden bg-transparent">
      <div className="hero-atmosphere pointer-events-none absolute inset-0" aria-hidden="true" />

      <div className="relative z-10 mx-auto flex min-h-[calc(100svh-76px)] w-full max-w-[1440px] flex-col items-center px-5 pb-10 pt-16 text-center sm:px-8 sm:pb-12 sm:pt-20 lg:pt-24">
        <div className="inline-flex items-center gap-2 rounded-full border border-dmg-accent/25 bg-dmg-accent-soft/70 px-4 py-2 text-xs font-bold uppercase tracking-[0.16em] text-dmg-accent">
          <span className="h-2 w-2 rounded-full bg-dmg-indicator" />
          Distributed LLM infrastructure
        </div>

        <div className="mt-8 max-w-5xl">
          <h1 className="font-display text-5xl font-semibold leading-[0.98] tracking-[-0.055em] text-dmg-text sm:text-7xl lg:text-8xl">
            <span className="block">Inference,</span>
            <span className="mt-1 block text-dmg-accent">distributed.</span>
          </h1>
          <p className="mx-auto mt-7 max-w-3xl text-base leading-7 text-dmg-muted sm:text-xl sm:leading-8">
            Orchestrate language model workloads across a distributed memory grid. Monitor your
            cluster and prepare inference requests from one workspace.
          </p>
          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            <Link
              to="/inference"
              className="inline-flex items-center gap-2 rounded-xl bg-dmg-accent px-6 py-3.5 text-base font-semibold text-dmg-accent-text shadow-lg shadow-dmg-accent/20 transition-colors hover:bg-dmg-accent-hover"
            >
              Open inference <ArrowRight size={18} />
            </Link>
            <Link
              to="/workers"
              className="inline-flex items-center gap-2 rounded-xl border border-dmg-line/25 bg-dmg-surface/75 px-6 py-3.5 text-base font-semibold text-dmg-text backdrop-blur transition-colors hover:bg-dmg-accent-soft"
            >
              Explore workers <ArrowDownRight size={18} />
            </Link>
          </div>
        </div>

        <section
          aria-label="Current platform status"
          className="mt-16 grid w-full max-w-5xl border-y border-dmg-line/25 bg-dmg-surface/65 text-left shadow-lg shadow-dmg-accent/5 backdrop-blur-md sm:grid-cols-3 sm:divide-x sm:divide-dmg-line/20 sm:mt-auto sm:pt-16"
        >
          {statusItems.map(({ label, value, detail, icon: Icon }) => (
            <article key={label} className="border-b border-dmg-line/20 px-5 py-5 last:border-b-0 sm:border-b-0 sm:px-7 sm:py-6">
              <div className="flex items-center gap-2.5 text-dmg-accent">
                <Icon size={17} strokeWidth={1.8} />
                <h2 className="text-sm font-bold uppercase tracking-[0.12em]">{label}</h2>
              </div>
              <p className="mt-4 font-display text-2xl font-semibold tracking-tight text-dmg-text sm:text-3xl">
                {value}
              </p>
              <p className="mt-1.5 text-sm leading-6 text-dmg-muted">{detail}</p>
            </article>
          ))}
        </section>
      </div>
    </section>
  );
}
