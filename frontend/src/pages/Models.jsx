import { Boxes } from "lucide-react";
import PageHeader from "../components/common/PageHeader";

export default function Models() {
  return (
    <>
      <PageHeader
        eyebrow="Inference resources"
        title="Models"
        description="Review the model catalog configured for your inference cluster."
      />
      <section className="grid min-h-72 place-items-center border border-dashed border-dmg-line/20 bg-dmg-surface/75 px-6 py-12 text-center backdrop-blur-md">
        <div className="max-w-sm">
          <span className="mx-auto grid h-14 w-14 place-items-center rounded-xl border border-dmg-line/20 bg-dmg-accent-soft text-dmg-accent">
            <Boxes size={24} strokeWidth={1.7} />
          </span>
          <h2 className="mt-6 text-lg font-semibold text-dmg-text">Model catalog not connected</h2>
          <p className="mt-3 text-base leading-7 text-dmg-muted">
            Model names, availability, and configuration will be shown after backend integration.
            No model data is being displayed yet.
          </p>
        </div>
      </section>
    </>
  );
}
