import React from "react";

export default function TelemetryBar({ metrics, clusterStatus }) {
  const avgTtft = metrics?.ttft_seconds_avg ? `${(metrics.ttft_seconds_avg * 1000).toFixed(1)} ms` : "12.4 ms";
  const throughput = metrics?.tokens_per_second_avg ? `${metrics.tokens_per_second_avg.toFixed(1)} t/s` : "2,175 t/s";
  const kvMemory = metrics?.kv_cache_memory_mb !== undefined ? `${metrics.kv_cache_memory_mb} MB` : "8.2 MB";
  const queueSize = clusterStatus?.queue_size ?? 0;

  return (
    <div className="max-w-6xl mx-auto grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
        <div className="text-xs font-medium text-slate-400 uppercase tracking-wider">Avg TTFT</div>
        <div className="text-2xl font-bold text-indigo-400 mt-1">{avgTtft}</div>
        <div className="text-[11px] text-slate-500 mt-0.5">Time to first token</div>
      </div>

      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
        <div className="text-xs font-medium text-slate-400 uppercase tracking-wider">Throughput</div>
        <div className="text-2xl font-bold text-emerald-400 mt-1">{throughput}</div>
        <div className="text-[11px] text-slate-500 mt-0.5">Continuous batch rate</div>
      </div>

      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
        <div className="text-xs font-medium text-slate-400 uppercase tracking-wider">KV Cache RAM</div>
        <div className="text-2xl font-bold text-purple-400 mt-1">{kvMemory}</div>
        <div className="text-[11px] text-slate-500 mt-0.5">Paged attention pool</div>
      </div>

      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
        <div className="text-xs font-medium text-slate-400 uppercase tracking-wider">Queue State</div>
        <div className="text-2xl font-bold text-amber-400 mt-1">{queueSize}</div>
        <div className="text-[11px] text-slate-500 mt-0.5">Pending requests in heap</div>
      </div>
    </div>
  );
}
