import React from "react";
import ModelControls from "./ModelControls";

export default function ChatConsole({
  prompt,
  setPrompt,
  streamedOutput,
  isGenerating,
  lastStats,
  outputRef,
  onSend,
  onCancel,
  temperature,
  setTemperature,
  maxTokens,
  setMaxTokens,
  priority,
  setPriority,
}) {
  return (
    <section className="lg:col-span-2 bg-slate-900/80 rounded-xl p-5 border border-slate-800 flex flex-col justify-between">
      <div>
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">Inference Console</h2>
          {lastStats && (
            <div className="text-xs text-slate-400 flex gap-3 font-mono">
              <span>Tokens: <strong className="text-white">{lastStats.tokens}</strong></span>
              <span>TTFT: <strong className="text-indigo-400">{(lastStats.ttft * 1000).toFixed(1)}ms</strong></span>
              <span>Speed: <strong className="text-emerald-400">{lastStats.tps.toFixed(1)} t/s</strong></span>
            </div>
          )}
        </div>

        <div
          ref={outputRef}
          className="min-h-[240px] max-h-[360px] overflow-y-auto bg-slate-950 p-4 rounded-lg text-sm font-mono border border-slate-800/90 whitespace-pre-wrap leading-relaxed shadow-inner"
        >
          {streamedOutput || (
            <span className="text-slate-600">Generated tokens stream here in real time...</span>
          )}
        </div>
      </div>

      <div className="mt-4 space-y-3">
        <ModelControls
          temperature={temperature}
          setTemperature={setTemperature}
          maxTokens={maxTokens}
          setMaxTokens={setMaxTokens}
          priority={priority}
          setPriority={setPriority}
        />

        <div className="flex gap-2">
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter prompt (e.g., 'Distributed memory systems achieve high throughput...')"
            className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-500"
            onKeyDown={(e) => e.key === "Enter" && onSend()}
          />
          {isGenerating ? (
            <button
              onClick={onCancel}
              className="bg-rose-600 hover:bg-rose-500 text-white font-medium text-sm px-6 py-2.5 rounded-lg transition-colors shadow-lg shadow-rose-600/20"
            >
              Stop
            </button>
          ) : (
            <button
              onClick={onSend}
              disabled={!prompt.trim()}
              className="bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-sm px-6 py-2.5 rounded-lg disabled:opacity-50 transition-colors shadow-lg shadow-indigo-600/20"
            >
              Generate
            </button>
          )}
        </div>
      </div>
    </section>
  );
}

