import React, { useState, useEffect, useRef } from "react";
import Navbar from "./components/Navbar";
import TelemetryBar from "./components/TelemetryBar";
import WorkerTopology from "./components/WorkerTopology";
import ChatConsole from "./components/ChatConsole";
import RequestHistory from "./components/RequestHistory";

export default function App() {
  const [workers, setWorkers] = useState([]);
  const [clusterStatus, setClusterStatus] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [history, setHistory] = useState([]);

  // Generation state
  const [prompt, setPrompt] = useState("");
  const [streamedOutput, setStreamedOutput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [activeJobId, setActiveJobId] = useState(null);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(25);
  const [priority, setPriority] = useState(1);
  const [lastStats, setLastStats] = useState(null);

  const outputRef = useRef(null);
  const wsRef = useRef(null);

  const fetchData = async () => {
    try {
      const [wRes, sRes, mRes, hRes] = await Promise.all([
        fetch("http://localhost:8000/api/workers").then((r) => r.json()),
        fetch("http://localhost:8000/api/cluster/status").then((r) => r.json()),
        fetch("http://localhost:8000/api/metrics").then((r) => r.json()),
        fetch("http://localhost:8000/api/inference/history").then((r) => r.json()),
      ]);
      setWorkers(wRes.workers || []);
      setClusterStatus(sRes);
      setMetrics(mRes);
      setHistory(hRes.history || []);
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
      setActiveJobId(data.job_id);

      const wsUrl = data.websocket_url?.startsWith("/")
        ? `ws://localhost:8000${data.websocket_url}`
        : `ws://localhost:8000/ws/inference/${data.job_id}`;

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.event === "token") {
          setStreamedOutput((prev) => prev + msg.token);
          if (outputRef.current) {
            outputRef.current.scrollTop = outputRef.current.scrollHeight;
          }
        } else if (msg.event === "done") {
          setIsGenerating(false);
          setActiveJobId(null);
          setLastStats({
            tokens: msg.tokens_generated,
            ttft: msg.ttft_seconds,
            tps: msg.tokens_per_second,
            totalTime: msg.total_latency_seconds,
          });
          fetchData();
          ws.close();
        } else if (msg.event === "cancelled") {
          setIsGenerating(false);
          setActiveJobId(null);
          fetchData();
          ws.close();
        }
      };

      ws.onerror = () => {
        setIsGenerating(false);
        setActiveJobId(null);
      };
    } catch (err) {
      console.error(err);
      setIsGenerating(false);
      setActiveJobId(null);
    }
  };

  const handleCancel = async () => {
    if (!activeJobId) return;
    try {
      await fetch(`http://localhost:8000/api/inference/cancel/${activeJobId}`, {
        method: "POST",
      });
      if (wsRef.current) {
        wsRef.current.close();
      }
      setIsGenerating(false);
      setActiveJobId(null);
      fetchData();
    } catch (e) {
      console.error(e);
      setIsGenerating(false);
      setActiveJobId(null);
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
          onCancel={handleCancel}
          temperature={temperature}
          setTemperature={setTemperature}
          maxTokens={maxTokens}
          setMaxTokens={setMaxTokens}
          priority={priority}
          setPriority={setPriority}
        />
      </main>

      <RequestHistory history={history} />
    </div>
  );
}
