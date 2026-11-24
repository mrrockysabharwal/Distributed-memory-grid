import React from "react";

export default function ModelControls({
  temperature,
  setTemperature,
  maxTokens,
  setMaxTokens,
  priority,
  setPriority,
}) {
  return (
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
  );
}

