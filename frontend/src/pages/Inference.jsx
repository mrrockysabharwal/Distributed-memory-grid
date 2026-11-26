import { useState } from "react";
import { AlertCircle, ArrowUp, Braces, Sparkles } from "lucide-react";
import PageHeader from "../components/common/PageHeader";

export default function Inference() {
  const [prompt, setPrompt] = useState("");
  const [notice, setNotice] = useState("");

  function handleSubmit(event) {
    event.preventDefault();
    if (!prompt.trim()) {
      setNotice("Enter a prompt before preparing an inference request.");
      return;
    }
    setNotice("Inference is not connected yet. Your prompt has not been sent.");
  }

  return (
    <>
      <PageHeader
        eyebrow="Generation"
        title="Inference"
        description="Prepare a prompt for distributed inference. Requests are not sent in this foundation build."
      />

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1.6fr)_minmax(280px,0.8fr)]">
        <section className="overflow-hidden border border-dmg-line/15 bg-dmg-surface/75 backdrop-blur-md">
          <div className="flex items-center gap-3 border-b border-dmg-line/15 px-5 py-4">
            <span className="grid h-10 w-10 place-items-center rounded-lg bg-dmg-accent-soft text-dmg-accent">
              <Sparkles size={18} />
            </span>
            <div>
              <h2 className="text-base font-semibold text-dmg-text">Inference console</h2>
              <p className="mt-1 text-sm text-dmg-muted">Prompt workspace</p>
            </div>
          </div>

          <div
            aria-live="polite"
            className="m-5 flex min-h-64 items-center justify-center border border-dmg-line/10 bg-dmg-surface-alt/75 px-6 py-8 text-center backdrop-blur"
          >
            <div className="max-w-sm">
              <Braces size={24} className="mx-auto text-dmg-accent" />
              <p className="mt-3 text-base font-medium text-dmg-text">Generated output will appear here</p>
              <p className="mt-2 text-sm leading-6 text-dmg-muted">
                Connect the inference backend to stream model responses.
              </p>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="px-5 pb-5">
            <label htmlFor="inference-prompt" className="mb-2 block text-sm font-semibold text-dmg-text">
              Prompt
            </label>
            <div className="flex flex-col gap-3 border border-dmg-line/20 bg-dmg-surface p-3 focus-within:border-dmg-accent/70 sm:flex-row sm:items-end">
              <textarea
                id="inference-prompt"
                value={prompt}
                onChange={(event) => {
                  setPrompt(event.target.value);
                  setNotice("");
                }}
                rows={3}
                placeholder="Write a prompt to prepare an inference request..."
                className="min-h-28 flex-1 resize-y bg-transparent px-1 py-1 text-base leading-7 text-dmg-text outline-none placeholder:text-dmg-subtle"
              />
              <button
                type="submit"
                className="inline-flex shrink-0 items-center justify-center gap-2 rounded-lg bg-dmg-accent px-5 py-3 text-base font-semibold text-dmg-accent-text transition-colors hover:bg-dmg-accent-hover"
              >
                <ArrowUp size={16} />
                Prepare request
              </button>
            </div>
            {notice && (
              <p role="status" className="mt-3 flex items-start gap-2 text-sm leading-6 text-dmg-accent">
                <AlertCircle size={14} className="mt-0.5 shrink-0" />
                {notice}
              </p>
            )}
          </form>
        </section>

        <aside className="h-fit border border-dmg-line/15 bg-dmg-surface/75 p-5 backdrop-blur-md">
          <h2 className="text-base font-semibold text-dmg-text">Request details</h2>
          <p className="mt-2 text-sm leading-6 text-dmg-muted">Configuration will be available when connected.</p>
          <dl className="mt-6 divide-y divide-dmg-line/15">
            <div className="flex items-center justify-between py-3">
              <dt className="text-sm text-dmg-muted">Model</dt>
              <dd className="text-sm font-medium text-dmg-text">Not selected</dd>
            </div>
            <div className="flex items-center justify-between py-3">
              <dt className="text-sm text-dmg-muted">Execution</dt>
              <dd className="text-sm font-medium text-dmg-text">Distributed</dd>
            </div>
            <div className="flex items-center justify-between py-3">
              <dt className="text-sm text-dmg-muted">Connection</dt>
              <dd className="text-sm font-medium text-dmg-text">Not connected</dd>
            </div>
          </dl>
          <p className="mt-4 border-l-2 border-dmg-accent/50 bg-dmg-accent-soft/60 p-4 text-sm leading-6 text-dmg-muted">
            Prompt text stays in this page and is not sent to a service.
          </p>
        </aside>
      </div>
    </>
  );
}
