import React, { useState, useEffect, useRef } from "react";
import Navbar from "./components/Navbar";
import TelemetryBar from "./components/TelemetryBar";
import WorkerTopology from "./components/WorkerTopology";
import ChatConsole from "./components/ChatConsole";

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

  const handleRebalance = async () => {
    try {
      await fetch("http://localhost:8000/api/models/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: "distilgpt2", total_layers: 12 }),
      });
      fetchData();
    } catch (e) {}
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 font-sans">
      <Navbar clusterStatus={clusterStatus} workerCount={workers.length} />
      <TelemetryBar metrics={metrics} clusterStatus={clusterStatus} />

      <main className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6">
        <WorkerTopology workers={workers} onRebalance={handleRebalance} />
        <ChatConsole
          prompt={prompt}
          setPrompt={setPrompt}
          streamedOutput={streamedOutput}
          isGenerating={isGenerating}
          lastStats={lastStats}
          outputRef={outputRef}
          onSend={handleSendPrompt}
          temperature={temperature}
          setTemperature={setTemperature}
          maxTokens={maxTokens}
          setMaxTokens={setMaxTokens}
          priority={priority}
          setPriority={setPriority}
        />
      </main>
    </div>
  );
}
