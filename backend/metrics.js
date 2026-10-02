const client = require('prom-client');

const register = new client.Registry();

client.collectDefaultMetrics({ register });

// ─── HTTP metrics ───────────────────────────────────────────────────────────
const httpRequestsTotal = new client.Counter({
  name: 'http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'route', 'status_code'],
  registers: [register],
});

const httpRequestDuration = new client.Histogram({
  name: 'http_request_duration_seconds',
  help: 'HTTP request duration in seconds',
  labelNames: ['method', 'route', 'status_code'],
  registers: [register],
  buckets: [0.1, 0.3, 0.5, 1, 2, 5],
});

// ─── AI / ML metrics ────────────────────────────────────────────────────────
const predictionCounter = new client.Counter({
  name: 'model_predictions_total',
  help: 'Total number of model predictions',
  registers: [register],
});

const predictionFailuresCounter = new client.Counter({
  name: 'model_predictions_failed_total',
  help: 'Total number of failed model predictions',
  registers: [register],
});

const predictionDuration = new client.Histogram({
  name: 'model_prediction_duration_seconds',
  help: 'Model prediction duration',
  registers: [register],
  buckets: [0.1, 0.5, 1, 2, 5, 10],
});

const guardrailCounter = new client.Counter({
  name: 'mri_guardrail_rejections_total',
  help: 'MRI images rejected by guardrail',
  registers: [register],
});

module.exports = {
  register,
  httpRequestsTotal,
  httpRequestDuration,
  predictionCounter,
  predictionFailuresCounter,
  predictionDuration,
  guardrailCounter,
};