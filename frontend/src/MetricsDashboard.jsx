import { useState, useEffect, useRef, useCallback } from 'react';
import './metrics.css';

/* ── Formatting helpers ─────────────────────────────────────────── */
const fmt = (v, digits = 1) =>
  v === null || v === undefined || !Number.isFinite(v) ? '—' : Number(v).toFixed(digits);

const fmtMs = (v) =>
  Number.isFinite(v) ? `${v < 100 ? v.toFixed(1) : v.toFixed(0)} ms` : '—';

const fmtBytes = (b) => {
  if (!Number.isFinite(b) || b <= 0) return '—';
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let i = 0;
  while (b >= 1024 && i < units.length - 1) { b /= 1024; i += 1; }
  return `${b.toFixed(b >= 100 ? 0 : 1)} ${units[i]}`;
};

const fmtUptime = (s) => {
  if (!Number.isFinite(s)) return '—';
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  return d > 0 ? `${d}d ${h}h` : `${h}h ${Math.floor((s % 3600) / 60)}m`;
};

const pct = (v) => (Number.isFinite(v) ? `${fmt(v)}%` : '—');

const targetState = (state) => (state === true ? 'pill-green' : state === false ? 'pill-red' : 'dim');

/* ── Sparkline (subtle SVG line in the design-system colours) ───── */
function Sparkline({ data, color = 'cyan', height = 54 }) {
  const points = (data || []).filter((p) => Number.isFinite(p[1]));
  if (points.length < 2) {
    return (
      <svg viewBox="0 0 300 40" preserveAspectRatio="none" style={{ height }}>
        <line x1="0" y1="20" x2="300" y2="20" stroke="rgba(255,255,255,0.08)" strokeWidth="1" strokeDasharray="3 4" />
      </svg>
    );
  }
  const values = points.map((p) => p[1]);
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) { min -= 1; max += 1; }
  const range = max - min;
  const W = 300, H = 40, pad = 3;
  const coords = points.map((p, i) => {
    const x = pad + (i / (points.length - 1)) * (W - pad * 2);
    const y = H - pad - ((p[1] - min) / range) * (H - pad * 2);
    return [x, y];
  });
  const path = coords.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
  const [lx, ly] = coords[coords.length - 1];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ height }}>
      <path d={path} className={`pline ${color === 'violet' ? 'violet' : ''}`} />
      <circle cx={lx} cy={ly} r="2.4" fill={color === 'violet' ? 'var(--violet)' : 'var(--cyan)'} />
    </svg>
  );
}

/* ── Small building blocks ──────────────────────────────────────── */
const StatTile = ({ icon, label, value, sub }) => (
  <div className="m-stat">
    <div className="m-stat-lbl"><i>{icon}</i> {label}</div>
    <div className="m-stat-val">{value}</div>
    {sub && <div className="m-stat-sub">{sub}</div>}
  </div>
);

const UtilBar = ({ label, pctVal, cls = 'cpu', foot }) => (
  <div className="m-bar">
    <div className="m-bar-lbl"><span>{label}</span><span>{foot}</span></div>
    <div className="m-bar-track">
      <div className={`m-bar-fill ${cls} ${pctVal > 80 ? 'hot' : ''}`} style={{ width: `${Math.min(100, pctVal)}%` }} />
    </div>
  </div>
);

const SparkCard = ({ label, value, data, color }) => (
  <div className="m-spark">
    <div className="m-spark-lbl"><span>{label}</span><b>{value}</b></div>
    <Sparkline data={data} color={color} />
  </div>
);

const EMPTY = { ok: false, degraded: true };

/* ── Main dashboard ─────────────────────────────────────────────── */
export default function MetricsDashboard({ API, token }) {
  const [data, setData] = useState(EMPTY);
  const inFlight = useRef(false);

  const poll = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    try {
      const res = await fetch(`${API}/api/metrics`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
    } catch {
      setData({ ok: false, degraded: true });
    } finally {
      inFlight.current = false;
    }
  }, [API, token]);

  useEffect(() => {
    poll();
    const id = setInterval(poll, 5000);
    return () => clearInterval(id);
  }, [poll]);

  const d = data.degraded || !data.ok ? null : data;

  if (!d) {
    return (
      <div className="metrics-zone anim-fade-in">
        <div className="card">
          <div className="card-hd">
            <div className="card-ico ico-blue">📊</div>
            <span className="card-title">System Monitoring</span>
            <span className="card-chip chip-blue">offline</span>
          </div>
          <div className="card-body">
            <div className="err-box">
              <div className="err-ico">⚠️</div>
              <div>
                <div className="err-ttl">Metrics unavailable</div>
                <div className="err-msg">
                  Could not reach the metrics service. Retrying automatically every 5 seconds.
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const lastRate = (series) => (series && series.length ? series[series.length - 1][1] : null);
  const reqRateNow = lastRate(d.spark.requests);
  const targetsUp = [d.targets.backend, d.targets.nodeExporter, d.targets.cadvisor]
    .filter((t) => t === true).length;
  const memFoot = `${fmtBytes(d.node?.memUsedBytes)} / ${fmtBytes(d.node?.memTotalBytes)}`;
  const containerList = (d.containers || [])
    .map((c) => c.name.replace(/^neurosight-/, ''))
    .join(', ');

  return (
    <div className="metrics-zone anim-fade-in">
      <div className="card">
        <div className="card-hd">
          <div className="card-ico ico-blue">📊</div>
          <span className="card-title">System Monitoring</span>
          <span className="card-chip chip-green">{targetsUp}/3 targets up</span>
          <span className="card-chip chip-violet">updates every 5s</span>
        </div>

        <div className="card-body">
          <div className="m-status">
            <span className={`pill ${targetState(d.targets.backend)}`}><span className="status-dot" /> backend</span>
            <span className={`pill ${targetState(d.targets.nodeExporter)}`}><span className="status-dot" /> node-exporter</span>
            <span className={`pill ${targetState(d.targets.cadvisor)}`}><span className="status-dot" /> cadvisor</span>
            <span className="pill pill-violet">uptime {fmtUptime(d.node?.uptimeSeconds)}</span>
          </div>

          <div className="m-tiles">
            <StatTile icon="⚡" label="Request rate" value={`${fmt(reqRateNow, 2)}/s`} sub={`total ${fmt(d.http?.total, 0)} requests`} />
            <StatTile icon="🕒" label="Latency p95" value={fmtMs(d.http?.p95s * 1000)} sub={<>p50 <em>{fmtMs(d.http?.p50s * 1000)}</em> · p99 <em>{fmtMs(d.http?.p99s * 1000)}</em></>} />
            <StatTile icon="🧠" label="Predictions" value={fmt(d.predictions?.total, 0)} sub={<>rate <em>{fmt(d.predictions?.rate, 2)}</em>/s</>} />
            <StatTile icon="🛡" label="Guardrail" value={fmt(d.predictions?.guardrail, 0)} sub={<>failed <em className="amber">{fmt(d.predictions?.failed, 0)}</em></>} />
            <StatTile icon="📈" label="Errors 4xx" value={fmt(d.http?.err4xxRps, 2)} sub={(<>5xx <em>{fmt(d.http?.err5xxRps, 2)}</em>/s</>)} />
            <StatTile icon="⏱" label="Predict time" value={fmtMs(d.predictions?.avgMs)} sub={<>p95 <em>{fmtMs(d.predictions?.p95Ms)}</em> · p99 <em>{fmtMs(d.predictions?.p99Ms)}</em></>} />
            <StatTile icon="💾" label="Disk usage" value={pct(d.node?.diskPct)} sub={<>{containerList || 'containers'} · backend <em>{fmtBytes(d.containers?.find(c => c.name === 'backend')?.memBytes)}</em></>} />
            <StatTile icon="✨" label="Failed preds" value={fmt(d.predictions?.failed, 0)} sub="model_predictions_failed_total" />
          </div>

          <div className="m-bars">
            <UtilBar label="CPU usage" pctVal={d.node?.cpuPct || 0} cls="cpu" foot={pct(d.node?.cpuPct)} />
            <UtilBar label="Memory usage" pctVal={d.node?.memPct || 0} cls="mem" foot={memFoot} />
          </div>

          <div className="m-sparks">
            <SparkCard label="Requests" value={`${fmt(reqRateNow, 2)}/s`} data={d.spark.requests} color="cyan" />
            <SparkCard label="Predictions" value={`${fmt(d.predictions?.rate, 2)}/s`} data={d.spark.predictions} color="violet" />
            <SparkCard label="CPU" value={pct(d.node?.cpuPct)} data={d.spark.cpu} color="cyan" />
            <SparkCard label="Memory" value={fmtBytes(d.node?.memUsedBytes)} data={d.spark.mem} color="violet" />
          </div>

          <div className="m-foot">
            <span className="ok">● system online — pulling from Prometheus every 5s</span>
            <span>last sync {new Date(d.fetchedAt).toLocaleTimeString()} · {containerList || 'no containers'}</span>
          </div>
        </div>
      </div>
    </div>
  );
}