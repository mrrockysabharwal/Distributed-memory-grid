import { ArrowUpRight } from "lucide-react";

export default function MetricCard({ label, value, detail }) {
  return (
    <article className="flex items-center justify-between gap-5 px-5 py-5 sm:px-7">
      <div>
        <p className="text-base font-semibold text-dmg-text">{label}</p>
        <p className="mt-1.5 text-sm leading-6 text-dmg-muted">{detail}</p>
      </div>
      <div className="flex shrink-0 items-center gap-4">
        <p className="text-2xl font-semibold tracking-tight text-dmg-accent">{value}</p>
        <ArrowUpRight size={17} className="text-dmg-accent" aria-hidden="true" />
      </div>
    </article>
  );
}
