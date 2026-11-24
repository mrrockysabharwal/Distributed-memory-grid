import React from "react";

export default function WorkerTopology({ workers, onRebalance }) {
  return (
    <section className="bg-slate-900/80 rounded-xl p-5 border border-slate-800 flex flex-col justify-between">
      <div>
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">Cluster Nodes</h2>
          <span className="text-xs bg-slate-800 px-2 py-0.5 rounded text-slate-400">{workers.length} Active</span>
        </div>

        <div className="space-y-3">
          {workers.map((w) => (
            <div key={w.worker_id} className="p-3.5 bg-slate-950/80 rounded-lg border border-slate-800/80 text-xs">
              <div className="flex justify-between items-center font-medium mb-1.5">
                <span className="text-slate-200 text-sm font-semibold">{w.worker_id}</span>
                <span className="text-[10px] px-2 py-0.5 bg-emerald-950/80 text-emerald-400 border border-emerald-800/60 rounded uppercase font-semibold">
                  {w.status}
                </span>
              </div>
              <div className="text-slate-400 mt-1">
                Partition: <span className="text-slate-200 font-mono font-medium">{w.model_partition}</span>
              </div>
              <div className="text-slate-400 mt-0.5">
                RAM Available: <span className="text-slate-200 font-mono">{w.available_memory_mb} MB</span>
              </div>
              <div className="text-slate-400 mt-0.5">
                Address: <span className="text-slate-400 font-mono text-[11px]">{w.address}</span>
              </div>
            </div>
          ))}
          {workers.length === 0 && (
            <p className="text-xs text-slate-500 italic p-3">Scanning for registered workers...</p>
          )}
        </div>
      </div>

      <div className="pt-4 border-t border-slate-800/80 mt-4">
        <button
          onClick={onRebalance}
          className="w-full text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium py-2 rounded-lg transition-colors border border-slate-700"
        >
          🔄 Rebalance 12 Layer Partitions
        </button>
      </div>
    </section>
  );
}

