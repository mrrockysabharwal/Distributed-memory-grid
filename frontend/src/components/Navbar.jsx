import React from "react";

export default function Navbar({ clusterStatus, workerCount }) {
  const isHealthy = clusterStatus?.cluster_state === "healthy";

  return (
    <header className="max-w-6xl mx-auto border-b border-slate-800 pb-4 mb-6 flex flex-wrap justify-between items-center gap-4">
      <div>
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center font-bold text-white text-base shadow-lg shadow-indigo-500/20">
            DMG
          </div>
          <h1 className="text-xl font-bold tracking-tight text-white">Distributed Memory Grid</h1>
        </div>
        <p className="text-xs text-slate-400 mt-1">Real-Time Continuous Batching & Pipeline Parallel LLM Inference</p>
      </div>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-full text-xs font-medium">
          <span className={`inline-block w-2.5 h-2.5 rounded-full ${isHealthy ? "bg-emerald-500 animate-pulse" : "bg-amber-500"}`}></span>
          <span className="capitalize">{clusterStatus?.cluster_state || "Online"}</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-full text-xs text-slate-400">
          Workers: <span className="text-white font-semibold">{workerCount}</span>
        </div>
      </div>
    </header>
  );
}
