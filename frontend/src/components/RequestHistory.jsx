import React from "react";

export default function RequestHistory({ history }) {
  if (!history || history.length === 0) {
    return null;
  }

  return (
    <div className="max-w-6xl mx-auto mt-6 bg-slate-900/80 rounded-xl p-5 border border-slate-800">
      <div className="flex justify-between items-center mb-3">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">Recent Request History</h2>
        <span className="text-xs text-slate-500">{history.length} Jobs Logged</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs font-mono">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400">
              <th className="pb-2">Job ID</th>
              <th className="pb-2">Status</th>
              <th className="pb-2">Prompt</th>
              <th className="pb-2">Tokens</th>
              <th className="pb-2">TTFT</th>
              <th className="pb-2">Throughput</th>
              <th className="pb-2">Latency</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {history.map((item) => (
              <tr key={item.job_id} className="hover:bg-slate-950/40">
                <td className="py-2.5 text-slate-400 font-semibold">{item.job_id.slice(0, 8)}...</td>
                <td className="py-2.5">
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] uppercase font-semibold border ${
                      item.status === "completed"
                        ? "bg-emerald-950/80 text-emerald-400 border-emerald-800/60"
                        : "bg-red-950/80 text-red-400 border-red-800/60"
                    }`}
                  >
                    {item.status}
                  </span>
                </td>
                <td className="py-2.5 text-slate-300 max-w-xs truncate">{item.prompt}</td>
                <td className="py-2.5 text-white">{item.tokens_generated}</td>
                <td className="py-2.5 text-indigo-400">{(item.ttft_seconds * 1000).toFixed(1)}ms</td>
                <td className="py-2.5 text-emerald-400">{item.tokens_per_second.toFixed(1)} t/s</td>
                <td className="py-2.5 text-slate-400">{item.total_latency_seconds.toFixed(2)}s</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
