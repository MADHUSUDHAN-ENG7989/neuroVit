// Prometheus dashboard payload builder.
// Fetches live metrics from Prometheus (docker service name `prometheus`),
// using the same PromQL the provisioned Grafana dashboard uses.

const PROMETHEUS_URL = process.env.PROMETHEUS_URL || 'http://prometheus:9090';
const REQUEST_TIMEOUT_MS = 3000;
const SPARK_WINDOW_S = 1800; // last 30 minutes
const SPARK_STEP_S = 30;

function numOrNull(v) {
  const n = parseFloat(v);
  return Number.isFinite(n) ? n : null;
}

async function fetchJson(url) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), REQUEST_TIMEOUT_MS);
  try {
    const res = await fetch(url, { signal: ctrl.signal });
    if (!res.ok) return null;
    const data = await res.json();
    return data && data.status === 'success' ? data.data : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

// Instant query → array of series (each { metric, value })
async function query(promql) {
  const data = await fetchJson(`${PROMETHEUS_URL}/api/v1/query?query=${encodeURIComponent(promql)}`);
  return data ? data.result : null;
}

// Range query → flattened series as [[epochMs, value], ...]
async function queryRange(promql, seconds = SPARK_WINDOW_S, step = SPARK_STEP_S) {
  const now = Math.floor(Date.now() / 1000);
  const start = now - seconds;
  const data = await fetchJson(
    `${PROMETHEUS_URL}/api/v1/query_range?query=${encodeURIComponent(promql)}&start=${start}&end=${now}&step=${step}`
  );
  if (!data || !data.result || !data.result.length) return [];
  return (data.result[0].values || []).map(([ts, v]) => [ts * 1000, numOrNull(v)]);
}

function sumSeries(result) {
  if (!result || !result.length) return null;
  let total = 0;
  let any = false;
  for (const s of result) {
    const v = parseFloat(Array.isArray(s.value) ? s.value[1] : s.value);
    if (Number.isFinite(v)) {
      total += v;
      any = true;
    }
  }
  return any ? total : null;
}

function scalar(result) {
  if (!result || !result.length) return null;
  const v = parseFloat(Array.isArray(result[0].value) ? result[0].value[1] : result[0].value);
  return numOrNull(v);
}

async function quantile(q, bucketPromql) {
  return scalar(await query(`histogram_quantile(${q}, sum(rate(${bucketPromql}[5m])) by (le))`));
}

function mapByLabel(result, label, converter = (v) => v) {
  const out = new Map();
  for (const s of result || []) {
    const name = s.metric && s.metric[label];
    if (name) out.set(name, converter(parseFloat(Array.isArray(s.value) ? s.value[1] : s.value)));
  }
  return [...out.entries()].map(([name, value]) => ({ name, value }));
}

async function buildDashboardPayload() {
  // ── reachability probe (if this fails, everything is degraded) ──────
  const up = await query('up');
  if (!up) return { ok: false, degraded: true, fetchedAt: new Date().toISOString() };

  const targetStatus = {};
  for (const s of up) {
    const job = s.metric && s.metric.job;
    if (job) targetStatus[job] = parseFloat(s.value[1]) === 1;
  }

  const [
    sparkRequests, sparkPredictions, podCpu, podMem,
  ] = await Promise.all([
    queryRange('sum(rate(http_requests_total[5m]))'),
    queryRange('sum(rate(model_predictions_total[5m]))'),
    queryRange('100 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100'),
    queryRange('node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes'),
  ]);

  const lastPoint = (series) => series.length ? series[series.length - 1][1] : null;

  const [
    httpRps, httpTotal, p50, p95, p99, err4xx, err5xx,
    predTotal, predFailed, guardrail, predAvgMs, predP95Ms, predP99Ms,
    nodeCpu, nodeMemPct, nodeDiskPct, nodeUptime,
    containersCpu, containersMem,
  ] = await Promise.all([
    query('sum(rate(http_requests_total[5m]))'),
    query('sum(http_requests_total)'),
    quantile(0.5, 'http_request_duration_seconds_bucket'),
    quantile(0.95, 'http_request_duration_seconds_bucket'),
    quantile(0.99, 'http_request_duration_seconds_bucket'),
    query('sum(rate(http_requests_total{status_code=~"4.."}[5m]))'),
    query('sum(rate(http_requests_total{status_code=~"5.."}[5m]))'),
    query('sum(model_predictions_total)'),
    query('sum(model_predictions_failed_total)'),
    query('sum(mri_guardrail_rejections_total)'),
    query('rate(model_prediction_duration_seconds_sum[5m]) / rate(model_prediction_duration_seconds_count[5m])'),
    quantile(0.95, 'model_prediction_duration_seconds_bucket'),
    quantile(0.99, 'model_prediction_duration_seconds_bucket'),
    query('100 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100'),
    query('(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100'),
    query('100 - (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100'),
    query('time() - node_boot_time_seconds'),
    query('sum(rate(container_cpu_usage_seconds_total{container_label_com_docker_compose_service!=""}[5m])) by (container_label_com_docker_compose_service) * 100'),
    query('sum(container_memory_usage_bytes{container_label_com_docker_compose_service!=""}) by (container_label_com_docker_compose_service)'),
  ]);

  const nodeMemUsed = await query('node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes');
  const nodeMemTotal = await query('node_memory_MemTotal_bytes');

  const containersByName = new Map();
  for (const c of mapByLabel(containersCpu, 'container_label_com_docker_compose_service')) containersByName.set(c.name, { name: c.name, cpuPct: c.value });
  for (const c of mapByLabel(containersMem, 'container_label_com_docker_compose_service')) {
    const entry = containersByName.get(c.name) || { name: c.name, cpuPct: null };
    entry.memBytes = c.value;
    containersByName.set(c.name, entry);
  }

  return {
    ok: true,
    degraded: false,
    fetchedAt: new Date().toISOString(),
    targets: {
      backend: targetStatus['node-backend'] ?? null,
      nodeExporter: targetStatus['node-exporter'] ?? null,
      cadvisor: targetStatus['cadvisor'] ?? null,
    },
    http: {
      rps: scalar(httpRps),
      total: scalar(httpTotal),
      p50s: scalar(p50),
      p95s: scalar(p95),
      p99s: scalar(p99),
      err4xxRps: scalar(err4xx),
      err5xxRps: scalar(err5xx),
    },
    predictions: {
      total: scalar(predTotal),
      rate: lastPoint(sparkPredictions),
      failed: scalar(predFailed),
      guardrail: scalar(guardrail),
      avgMs: scalar(predAvgMs) != null ? scalar(predAvgMs) * 1000 : null,
      p95Ms: scalar(predP95Ms) != null ? scalar(predP95Ms) * 1000 : null,
      p99Ms: scalar(predP99Ms) != null ? scalar(predP99Ms) * 1000 : null,
    },
    node: {
      cpuPct: scalar(nodeCpu),
      memPct: scalar(nodeMemPct),
      memUsedBytes: scalar(nodeMemUsed),
      memTotalBytes: scalar(nodeMemTotal),
      diskPct: scalar(nodeDiskPct),
      uptimeSeconds: scalar(nodeUptime),
    },
    containers: [...containersByName.values()],
    spark: {
      requests: sparkRequests,
      predictions: sparkPredictions,
      cpu: podCpu,
      mem: podMem,
    },
  };
}

module.exports = { buildDashboardPayload, PROMETHEUS_URL };