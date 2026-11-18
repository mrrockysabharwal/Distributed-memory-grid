import React, { useState, useEffect } from "react";

export default function App() {
  const [workers, setWorkers] = useState([]);
  const [prompt, setPrompt] = useState("");
  const [streamedOutput, setStreamedOutput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);

  useEffect(() => {
    fetch("http://localhost:8000/api/workers")
      .then((res) => res.json())
      .then((data) => setWorkers(data.workers || []))
      .catch(() => {});
  }, []);

  const handleSendPrompt = async () => {
    if (!prompt.trim() || isGenerating) return;
    setIsGenerating(true);
    setStreamedOutput("");

    try {
      const res = await fetch("http://localhost:8000/api/inference/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, model_id: "distilgpt2" }),
      });
      const data = await res.json();
      
      const ws = new WebSocket(`ws://localhost:8000/ws/inference/${data.job_id}`);
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.event === "token") {
          setStreamedOutput((prev) => prev + msg.token);
        } else if (msg.event === "done") {
          setIsGenerating(false);
          ws.close();
        }
      };
      ws.onerror = () => setIsGenerating(false);
    } catch (err) {
      console.error(err);
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 p-8 font-sans">
      <header className="max-w-5xl mx-auto border-b border-slate-700 pb-4 mb-8 flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Distributed Memory Grid</h1>
          <p className="text-sm text-slate-400">Real-Time LLM Inference Orchestration</p>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-block w-3 h-3 bg-emerald-500 rounded-full animate-pulse"></span>
          <span className="text-xs uppercase font-semibold text-emerald-400">Cluster Online</span>
        </div>
      </header>

      <main className="max-w-5xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Left Column: Worker Nodes */}
        <section className="bg-slate-800/80 rounded-xl p-5 border border-slate-700">
          <h2 className="text-base font-semibold mb-4 text-slate-200">Worker Topology</h2>
          <div className="space-y-3">
            {workers.map((w) => (
              <div key={w.worker_id} className="p-3 bg-slate-900/60 rounded-lg border border-slate-700/60 text-xs">
                <div className="flex justify-between font-medium text-slate-300 mb-1">
                  <span>{w.worker_id}</span>
                  <span className="text-emerald-400">{w.status}</span>
                </div>
                <div className="text-slate-400">Partition: {w.model_partition}</div>
                <div className="text-slate-400">Memory: {w.available_memory_mb} MB</div>
              </div>
            ))}
            {workers.length === 0 && (
              <p className="text-xs text-slate-500">No workers currently detected.</p>
            )}
          </div>
        </section>

        {/* Right Column: Chat & Streaming */}
        <section className="md:col-span-2 bg-slate-800/80 rounded-xl p-5 border border-slate-700 flex flex-col justify-between">
          <div>
            <h2 className="text-base font-semibold mb-4 text-slate-200">Interactive Inference Console</h2>
            <div className="min-h-[220px] max-h-[350px] overflow-y-auto bg-slate-950 p-4 rounded-lg text-sm font-mono border border-slate-800 whitespace-pre-wrap">
              {streamedOutput || (
                <span className="text-slate-600">Generated tokens will stream here in real time...</span>
              )}
            </div>
          </div>

          <div className="mt-4 flex gap-2">
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter prompt (e.g., 'Once upon a time in a distributed system...')"
              className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-indigo-500"
              onKeyDown={(e) => e.key === "Enter" && handleSendPrompt()}
            />
            <button
              onClick={handleSendPrompt}
              disabled={isGenerating}
              className="bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-sm px-5 py-2 rounded-lg disabled:opacity-50 transition-colors"
            >
              {isGenerating ? "Streaming..." : "Generate"}
            </button>
          </div>
        </section>
      </main>
    </div>
  );
}

