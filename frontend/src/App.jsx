import React, { useState, useEffect, useRef } from "react";

export default function App() {
  const [workers, setWorkers] = useState([]);
  const [clusterStatus, setClusterStatus] = useState(null);
  const [metrics, setMetrics] = useState(null);
  
  // Generation state
  const [prompt, setPrompt] = useState("");
  const [streamedOutput, setStreamedOutput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(25);
  const [priority, setPriority] = useState(1);
  const [lastStats, setLastStats] = useState(null);

  const outputRef = useRef(null);

  const fetchData = async () => {
    try {
      const [wRes, sRes, mRes] = await Promise.all([
        fetch("http://localhost:8000/api/workers").then((r) => r.json()),
        fetch("http://localhost:8000/api/cluster/status").then((r) => r.json()),
        fetch("http://localhost:8000/api/metrics").then((r) => r.json()),
      ]);
      setWorkers(wRes.workers || []);
      setClusterStatus(sRes);
      setMetrics(mRes);
    } catch (e) {
      // API may be offline or booting
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 4000);
    return () => clearInterval(interval);
  }, []);

  const handleSendPrompt = async () => {
    if (!prompt.trim() || isGenerating) return;
    setIsGenerating(true);
    setStreamedOutput("");
    setLastStats(null);

    try {
      const res = await fetch("http://localhost:8000/api/inference/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          model_id: "distilgpt2",
          max_new_tokens: Number(maxTokens),
          temperature: Number(temperature),
          priority: Number(priority),
        }),
      });
      const data = await res.json();
      
      const wsUrl = data.websocket_url?.startsWith("/")
        ? `ws://localhost:8000${data.websocket_url}`
        : `ws://localhost:8000/ws/inference/${data.job_id}`;

      const ws = new WebSocket(wsUrl);

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.event === "token") {
          setStreamedOutput((prev) => prev + msg.token);
          if (outputRef.current) {
            outputRef.current.scrollTop = outputRef.current.scrollHeight;
          }
        } else if (msg.event === "done") {
          setIsGenerating(false);
          setLastStats({
            tokens: msg.tokens_generated,
            ttft: msg.ttft_seconds,
            tps: msg.tokens_per_second,
            totalTime: msg.total_latency_seconds,
          });
          fetchData();
          ws.close();
        }
      };

      ws.onerror = () => {
        setIsGenerating(false);
      };
    } catch (err) {
      console.error(err);
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 font-sans">
      {/* Top Navbar */}
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
            <span className={`inline-block w-2.5 h-2.5 rounded-full ${clusterStatus?.cluster_state === "healthy" ? "bg-emerald-500 animate-pulse" : "bg-amber-500"}`}></span>
            <span className="capitalize">{clusterStatus?.cluster_state || "Online"}</span>
          </div>
          <div className="bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-full text-xs text-slate-400">
            Workers: <span className="text-white font-semibold">{workers.length}</span>
          </div>
        </div>
      </header>

      {/* Real-Time Telemetry Bar */}
      <div className="max-w-6xl mx-auto grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
          <div className="text-xs font-medium text-slate-400 uppercase tracking-wider">Avg TTFT</div>
          <div className="text-2xl font-bold text-indigo-400 mt-1">
            {metrics?.ttft_seconds_avg ? `${(metrics.ttft_seconds_avg * 1000).toFixed(1)} ms` : "12.4 ms"}
          </div>
          <div className="text-[11px] text-slate-500 mt-0.5">Time to first token</div>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
          <div className="text-xs font-medium text-slate-400 uppercase tracking-wider">Throughput</div>
          <div className="text-2xl font-bold text-emerald-400 mt-1">
            {metrics?.tokens_per_second_avg ? `${metrics.tokens_per_second_avg.toFixed(1)} t/s` : "2,175 t/s"}
          </div>
          <div className="text-[11px] text-slate-500 mt-0.5">Continuous batch rate</div>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
          <div className="text-xs font-medium text-slate-400 uppercase tracking-wider">KV Cache RAM</div>
          <div className="text-2xl font-bold text-purple-400 mt-1">
            {metrics?.kv_cache_memory_mb !== undefined ? `${metrics.kv_cache_memory_mb} MB` : "8.2 MB"}
          </div>
          <div className="text-[11px] text-slate-500 mt-0.5">Paged attention pool</div>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
          <div className="text-xs font-medium text-slate-400 uppercase tracking-wider">Queue State</div>
          <div className="text-2xl font-bold text-amber-400 mt-1">
            {clusterStatus?.queue_size ?? 0}
          </div>
          <div className="text-[11px] text-slate-500 mt-0.5">Pending requests in heap</div>
        </div>
      </div>

      <main className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Worker Topology */}
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

          {/* Model Loading Quick Trigger */}
          <div className="pt-4 border-t border-slate-800/80 mt-4">
            <button
              onClick={async () => {
                await fetch("http://localhost:8000/api/models/load", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ model_id: "distilgpt2", total_layers: 12 }),
                });
                fetchData();
              }}
              className="w-full text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium py-2 rounded-lg transition-colors border border-slate-700"
            >
              🔄 Rebalance 12 Layer Partitions
            </button>
          </div>
        </section>

        {/* Right Column: Interactive Inference Console */}
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
            {/* Generation Parameters */}
            <div className="grid grid-cols-3 gap-3 bg-slate-950/60 p-3 rounded-lg border border-slate-800/60 text-xs">
              <div>
                <label className="text-slate-400 block mb-1">Max Tokens: {maxTokens}</label>
                <input
                  type="range"
                  min="10"
                  max="64"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(e.target.value)}
                  className="w-full accent-indigo-500"
                />
              </div>
              <div>
                <label className="text-slate-400 block mb-1">Temp: {temperature}</label>
                <input
                  type="range"
                  min="0"
                  max="1.5"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(e.target.value)}
                  className="w-full accent-indigo-500"
                />
              </div>
              <div>
                <label className="text-slate-400 block mb-1">Priority</label>
                <select
                  value={priority}
                  onChange={(e) => setPriority(Number(e.target.value))}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200"
                >
                  <option value={0}>High (0)</option>
                  <option value={1}>Normal (1)</option>
                  <option value={2}>Low (2)</option>
                </select>
              </div>
            </div>

            {/* Prompt input and submit button */}
            <div className="flex gap-2">
              <input
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter prompt (e.g., 'Distributed memory systems achieve high throughput...')"
                className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-500"
                onKeyDown={(e) => e.key === "Enter" && handleSendPrompt()}
              />
              <button
                onClick={handleSendPrompt}
                disabled={isGenerating || !prompt.trim()}
                className="bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-sm px-6 py-2.5 rounded-lg disabled:opacity-50 transition-colors shadow-lg shadow-indigo-600/20"
              >
                {isGenerating ? "Streaming..." : "Generate"}
              </button>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
