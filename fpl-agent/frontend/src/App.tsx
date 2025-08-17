import React, { useMemo, useState } from "react";

// ---- Types from backend ----
type Player = {
  name: string;
  pos: "GKP" | "DEF" | "MID" | "FWD";
  team: string;       // short code
  opponent: string;   // e.g. "FUL (H)"
  price: number;      // £m
  xPts: number;
};
type OptimizeResult = {
  captain: string | null;
  vice: string | null;
  starting_XI: Player[];
  bench: Player[];
  total_value: number;
  bank: number;
  objective_xpts: number;
};

// Same-origin by default to avoid CORS/preflight
const API_BASE = (import.meta as any).env?.VITE_API_BASE || "";

function clsx(...xs: Array<string | false | null | undefined>) {
  return xs.filter(Boolean).join(" ");
}

async function postJSON<T = any>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

const FORMATIONS = ["3-4-3", "3-5-2", "4-4-2", "4-3-3", "4-5-1", "5-4-1", "5-3-2", "5-2-3"];

function inferFormation(xi: Player[]) {
  const d = xi.filter((p) => p.pos === "DEF").length;
  const m = xi.filter((p) => p.pos === "MID").length;
  const f = xi.filter((p) => p.pos === "FWD").length;
  return { d, m, f };
}

function parseFreeText(message: string) {
  const gw = +(message.match(/gw[=\s]*(\d+)/i)?.[1] ?? 1);
  const formation = message.match(/(\d-\d-\d)/)?.[1] ?? "3-4-3";
  const budget = +(message.match(/(£|gbp|budget)[=\s]*(\d+\.?\d*)/i)?.[2] ?? 100.0);
  return { gw, formation, budget };
}

const StatPill: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="rounded-full bg-slate-800/80 border border-slate-700 px-3 py-1 text-xs text-slate-100">
    <span className="text-slate-300 mr-1">{label}:</span>
    <span className="font-semibold">{value}</span>
  </div>
);

const PlayerPill: React.FC<{ p: Player; highlight?: "C" | "VC" }> = ({ p, highlight }) => (
  <div
    className={clsx(
      "rounded-2xl px-3 py-2 border text-sm",
      "bg-slate-800/80 border-slate-700 text-slate-100 shadow",
      highlight === "C" && "ring-2 ring-emerald-400",
      highlight === "VC" && "ring-2 ring-cyan-400"
    )}
  >
    <div className="font-semibold">
      {p.name} {highlight ? `(${highlight})` : ""}
    </div>
    <div className="text-xs text-slate-300">
      {p.pos} – {p.team} • vs {p.opponent}
    </div>
    <div className="text-xs text-slate-300">
      £{p.price.toFixed(1)}m • xPts {p.xPts.toFixed(2)}
    </div>
  </div>
);

const Pitch: React.FC<{ xi: Player[]; captain: string | null; vice: string | null }> = ({
  xi,
  captain,
  vice,
}) => {
  const { d, m, f } = useMemo(() => inferFormation(xi), [xi]);
  const gk = xi.filter((p) => p.pos === "GKP");
  const def = xi.filter((p) => p.pos === "DEF").slice(0, d);
  const mid = xi.filter((p) => p.pos === "MID").slice(0, m);
  const fwd = xi.filter((p) => p.pos === "FWD").slice(0, f);

  const Row: React.FC<{ kids: Player[] }> = ({ kids }) => (
    <div className="flex flex-wrap justify-center gap-3">
      {kids.map((p, i) => (
        <PlayerPill
          key={p.name + i}
          p={p}
          highlight={p.name === captain ? "C" : p.name === vice ? "VC" : undefined}
        />
      ))}
    </div>
  );

  return (
    <div className="rounded-3xl border border-slate-700 bg-gradient-to-b from-slate-900 via-slate-900 to-slate-950 p-4 sm:p-6">
      <div className="text-center text-slate-200 text-xs mb-2">
        Formation: {d}-{m}-{f}
      </div>
      <div className="space-y-4 sm:space-y-5">
        <Row kids={fwd} />
        <Row kids={mid} />
        <Row kids={def} />
        <Row kids={gk} />
      </div>
    </div>
  );
};

const ListView: React.FC<{
  xi: Player[];
  bench: Player[];
  captain: string | null;
  vice: string | null;
}> = ({ xi, bench, captain, vice }) => (
  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
    <div>
      <div className="text-slate-100 font-semibold mb-2">Starting XI</div>
      <div className="space-y-2">
        {xi.map((p, i) => (
          <PlayerPill
            key={p.name + i}
            p={p}
            highlight={p.name === captain ? "C" : p.name === vice ? "VC" : undefined}
          />
        ))}
      </div>
    </div>
    <div>
      <div className="text-slate-100 font-semibold mb-2">Bench (GK + 3)</div>
      <div className="space-y-2">
        {bench.map((p, i) => (
          <PlayerPill key={p.name + i} p={p} />
        ))}
      </div>
    </div>
  </div>
);

export default function App() {
  // Controls
  const [gw, setGw] = useState<number>(1);
  const [formation, setFormation] = useState<string>("3-4-3");
  const [budget, setBudget] = useState<number>(100.0);
  const [formWeight, setFormWeight] = useState<number>(0.6); // how much to favor recent form vs PPG
  const [fdrWeight, setFdrWeight] = useState<number>(0.3);   // how much fixture difficulty affects xPts

  // Optional server LLM (Ollama)
  const [useServerLLM, setUseServerLLM] = useState<boolean>(false);
  const [agentMsg, setAgentMsg] = useState<string>(
    "What's the best FPL squad for GW3 with a £100m budget and a 4-4-2 formation?"
  );

  // Results & UI state
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<OptimizeResult | null>(null);
  const [mode, setMode] = useState<"pitch" | "list">("pitch");
  const [error, setError] = useState<string | null>(null);

  // API
  async function callOptimize(gwi: number, formi: string, bud: number) {
    setLoading(true);
    setError(null);
    try {
      const data = await postJSON<OptimizeResult>("/optimize", {
        gw: gwi,
        formation: formi,
        budget: bud,
        bench_weight: 0.1,
        form_weight: formWeight,
        fdr_weight: fdrWeight,
      });
      setResult(data);
    } catch (e: any) {
      setError(e?.message || "Network error");
    } finally {
      setLoading(false);
    }
  }

  async function onOptimize() {
    await callOptimize(gw, formation, budget);
  }

  async function onAskAgent() {
    setLoading(true);
    setError(null);
    try {
      const msg = agentMsg?.trim() || `Build me a team for gw=${gw}, ${formation}, £${budget}m`;
      if (useServerLLM) {
        await postJSON("/chat", { message: msg }).catch(() => null);
        const parsed = parseFreeText(msg);
        const data = await postJSON<OptimizeResult>("/optimize", {
          ...parsed,
          form_weight: formWeight,
          fdr_weight: fdrWeight,
        });
        setResult(data);
      } else {
        const parsed = parseFreeText(msg);
        const data = await postJSON<OptimizeResult>("/optimize", {
          ...parsed,
          form_weight: formWeight,
          fdr_weight: fdrWeight,
        });
        setResult(data);
      }
    } catch (e: any) {
      setError(e?.message || "Agent error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_top_left,rgba(67,56,202,0.18),transparent_60%),radial-gradient(ellipse_at_bottom_right,rgba(236,72,153,0.18),transparent_60%)] text-slate-100">
      <div className="max-w-5xl mx-auto px-4 pb-24">
        {/* Header */}
        <div className="py-8">
          <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight">
            <span className="bg-gradient-to-tr from-indigo-400 via-fuchsia-400 to-amber-300 bg-clip-text text-transparent drop-shadow-[0_1px_1px_rgba(0,0,0,0.8)]">
              FPL Free Agent
            </span>
          </h1>
          <p className="text-sm sm:text-base text-slate-200 mt-2">
            100% free: chatty agent + backend optimizer. Toggle server LLM or go direct.
          </p>
        </div>

        {/* Controls */}
        <div className="backdrop-blur-xl bg-slate-800/60 border border-slate-700 rounded-2xl p-4 sm:p-6 shadow-2xl">
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 items-end">
            <div>
              <label className="block text-xs text-slate-300 mb-1">Gameweek</label>
              <input
                type="number"
                min={1}
                className="w-full rounded-lg bg-slate-900/80 border border-slate-700 px-3 py-2 text-slate-100 placeholder-slate-400 focus:outline-none focus:ring focus:ring-indigo-500/40"
                value={gw}
                onChange={(e) => setGw(+e.target.value)}
              />
            </div>
            <div>
              <label className="block text-xs text-slate-300 mb-1">Formation</label>
              <select
                className="w-full rounded-lg bg-slate-900/80 border border-slate-700 px-3 py-2 text-slate-100 focus:outline-none focus:ring focus:ring-indigo-500/40"
                value={formation}
                onChange={(e) => setFormation(e.target.value)}
              >
                {FORMATIONS.map((f) => (
                  <option key={f} value={f}>
                    {f}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs text-slate-300 mb-1">Budget (£m)</label>
              <input
                type="number"
                step="0.1"
                min={80}
                max={200}
                className="w-full rounded-lg bg-slate-900/80 border border-slate-700 px-3 py-2 text-slate-100 placeholder-slate-400 focus:outline-none focus:ring focus:ring-amber-500/40"
                value={budget}
                onChange={(e) => setBudget(+e.target.value)}
              />
            </div>
            <div className="col-span-2 flex items-center justify-between gap-3 bg-slate-900/70 border border-slate-700 rounded-lg px-3 py-2">
              <div className="text-sm">
                <div className="text-slate-100 font-semibold">Server LLM (Ollama)</div>
                <div className="text-slate-300 text-xs">If off, we call /optimize directly.</div>
              </div>
              <button
                onClick={() => setUseServerLLM((v) => !v)}
                className={clsx(
                  "relative inline-flex h-8 w-16 items-center rounded-full transition",
                  useServerLLM ? "bg-emerald-500/80" : "bg-slate-700"
                )}
              >
                <span
                  className={clsx(
                    "inline-block h-6 w-6 transform rounded-full bg-white shadow transition",
                    useServerLLM ? "translate-x-8" : "translate-x-2"
                  )}
                />
              </button>
            </div>
          </div>

          {/* Weights */}
          <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="rounded-xl bg-slate-900/70 border border-slate-700 p-3">
              <div className="text-xs text-slate-300 mb-1">Form weight (vs. PPG)</div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={formWeight}
                onChange={(e) => setFormWeight(+e.target.value)}
                className="w-full"
              />
              <div className="text-sm text-slate-200 mt-1">{(formWeight * 100).toFixed(0)}%</div>
            </div>
            <div className="rounded-xl bg-slate-900/70 border border-slate-700 p-3">
              <div className="text-xs text-slate-300 mb-1">Fixture difficulty weight (FDR)</div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={fdrWeight}
                onChange={(e) => setFdrWeight(+e.target.value)}
                className="w-full"
              />
              <div className="text-sm text-slate-200 mt-1">{(fdrWeight * 100).toFixed(0)}%</div>
            </div>
          </div>

          {/* Actions */}
          <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-3">
            <button
              onClick={onOptimize}
              disabled={loading}
              className={clsx(
                "rounded-xl px-4 py-3 text-sm font-semibold shadow-lg",
                "bg-gradient-to-r from-indigo-500 via-fuchsia-500 to-amber-400",
                "hover:brightness-110 disabled:opacity-50 text-slate-100"
              )}
            >
              {loading ? "Optimizing…" : "Optimize with Controls"}
            </button>

            <div className="sm:col-span-2 flex gap-2">
              <input
                className="flex-1 rounded-xl bg-slate-900/80 border border-slate-700 px-3 py-3 text-slate-100 placeholder-slate-400 focus:outline-none focus:ring focus:ring-fuchsia-500/40"
                value={agentMsg}
                onChange={(e) => setAgentMsg(e.target.value)}
                placeholder="Ask the agent (e.g., Best squad for GW2, 4-4-2, £100m)"
              />
              <button
                onClick={onAskAgent}
                disabled={loading}
                className={clsx(
                  "rounded-xl px-4 py-3 text-sm font-semibold shadow-lg",
                  "bg-slate-800/70 border border-slate-700 hover:bg-slate-800",
                  loading && "opacity-50"
                )}
              >
                {loading ? "Thinking…" : "Ask Agent"}
              </button>
            </div>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="mt-6 rounded-xl border border-red-500/40 bg-red-500/10 text-red-200 px-4 py-3">
            {String(error)}
          </div>
        )}

        {/* Results */}
        {result ? (
          <div className="mt-6 backdrop-blur-xl bg-slate-900/90 border border-slate-700 rounded-2xl p-4 sm:p-6 shadow-2xl">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-slate-100 font-semibold tracking-wide">
                Captain: {result.captain ?? "-"} · Vice: {result.vice ?? "-"}
              </h2>
              <div className="flex items-center gap-2 text-xs">
                <button
                  onClick={() => setMode("pitch")}
                  className={clsx(
                    "px-3 py-1.5 rounded-full border",
                    mode === "pitch" ? "bg-slate-800/80 border-slate-700" : "bg-slate-900/60 border-slate-700"
                  )}
                >
                  Pitch
                </button>
                <button
                  onClick={() => setMode("list")}
                  className={clsx(
                    "px-3 py-1.5 rounded-full border",
                    mode === "list" ? "bg-slate-800/80 border-slate-700" : "bg-slate-900/60 border-slate-700"
                  )}
                >
                  List
                </button>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2 mb-4">
              <StatPill label="Total Value" value={`£${result.total_value.toFixed(1)}m`} />
              <StatPill label="Bank" value={`£${result.bank.toFixed(1)}m`} />
              <StatPill label="Objective xPts" value={result.objective_xpts.toFixed(2)} />
            </div>

            {mode === "pitch" ? (
              <Pitch xi={result.starting_XI} captain={result.captain} vice={result.vice} />
            ) : (
              <div className="space-y-4">
                <ListView
                  xi={result.starting_XI}
                  bench={result.bench}
                  captain={result.captain}
                  vice={result.vice}
                />
                <div>
                  <div className="text-slate-300 text-xs">Bench (GK + 3):</div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {result.bench.map((p, i) => (
                      <PlayerPill key={p.name + i} p={p} />
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="mt-10 text-center text-slate-300">Run an optimization to see your squad.</div>
        )}
      </div>
    </div>
  );
}
