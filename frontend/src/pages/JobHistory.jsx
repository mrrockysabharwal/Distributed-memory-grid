import { Clock3 } from "lucide-react";
import PageHeader from "../components/common/PageHeader";

export default function JobHistory() {
  return (
    <>
      <PageHeader
        eyebrow="Activity"
        title="Job history"
        description="Review inference requests and their execution status."
      />
      <section className="grid min-h-72 place-items-center border border-dashed border-dmg-line/20 bg-dmg-surface/75 px-6 py-12 text-center backdrop-blur-md">
        <div className="max-w-sm">
          <span className="mx-auto grid h-14 w-14 place-items-center rounded-xl border border-dmg-line/20 bg-dmg-accent-soft text-dmg-accent">
            <Clock3 size={24} strokeWidth={1.7} />
          </span>
          <h2 className="mt-6 text-lg font-semibold text-dmg-text">No jobs to show</h2>
          <p className="mt-3 text-base leading-7 text-dmg-muted">
            Job history will appear here after inference is connected. There are no recorded jobs
            in this frontend yet.
          </p>
        </div>
      </section>
    </>
  );
}
