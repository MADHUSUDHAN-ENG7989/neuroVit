# NeuroSight AI — Deploying to AWS EC2

This folder contains everything you need to run the pre-built Docker images on an EC2 instance.
You only need Docker on the EC2 box and the AWS credentials already used in your local `backend/.env`.

The `docker-compose.prod.yml` runs the app **plus a full monitoring stack**:

- `prometheus` (`:9090`) — scrapes the backend `/metrics`, host metrics, and container metrics
- `grafana` (`:3000`) — dashboards + alerts (Prometheus data source: `http://prometheus:9090`)
- `node-exporter` (`:9100`) — EC2 host CPU / RAM / disk / network
- `cadvisor` (`:8080`) — per-container CPU / RAM / network / filesystem

The backend exposes Prometheus metrics at `/metrics` (port 5000).

## 0. One-time AWS setup

- Create an EC2 instance (t2.medium / t3.medium recommended — the image is CPU-only).
- In the instance **Security Group** allow inbound: `HTTP 80` and `TCP 5000` from `0.0.0.0/0`.
- For monitoring, only expose `3000` (Grafana) if you want browser access; **limit it to your own IP**.
  Do **not** open `9090` / `9100` / `8080` to the public internet.
- Make sure your AWS IAM user has permissions to call DynamoDB (`AmazonDynamoDBFullAccess` is fine for demos).
- Create the DynamoDB tables (`Users`, `QueryLogs`) — see `backend/setup-dynamo.js`.

## 1. Get these files onto the instance

Clone the repo (lighter: copy only this folder):

```bash
scp -i your-key.pem -r deploy/ ec2-user@<PUBLIC_IP>:/home/ec2-user/neurosight
```

## 2. Configure

```bash
ssh -i your-key.pem ec2-user@<PUBLIC_IP>
cd ~/neurosight
cp .env.example .env
nano .env     # set AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, JWT_SECRET
```

## 3. Deploy

```bash
chmod +x deploy.sh
./deploy.sh
```

The script installs Docker, logs into Docker Hub, pulls `madhusudhanchilumula/neurosight-backend` and
`madhusudhanchilumula/neurosight-frontend`, and starts the stack.

## 4. Verify

- Open `http://<PUBLIC_IP>/` — the Frontend (nginx).
- Check the backend: `http://<PUBLIC_IP>:5000/api/health` → `{"status":"ok",...}`.
- Backend metrics: `http://<PUBLIC_IP>:5000/metrics` → Prometheus text.
- Prometheus: `http://<PUBLIC_IP>:9090` → Status → Targets (all `UP`).
- Grafana: `http://<PUBLIC_IP>:3000` (default login `admin`/`admin`, change it immediately).
- Logs: `bash docker compose -f docker-compose.prod.yml --env-file .env logs -f backend`

> The frontend talks to the backend **through nginx** (`/api` and `/uploads` are reverse-proxied),
> so no CORS / cross-origin config is needed from the browser.

## Grafana first-time setup

1. Connect → Data sources → Prometheus → set URL to `http://prometheus:9090` → **Save & test**.
2. Add panels with PromQL, e.g.:
   - Requests/s: `rate(http_requests_total[5m])`
   - 5xx rate: `rate(http_requests_total{status_code=~"5.."}[5m])`
   - Predictions: `model_predictions_total`
   - Avg inference: `rate(model_prediction_duration_seconds_sum[5m]) / rate(model_prediction_duration_seconds_count[5m])`
   - P95 inference: `histogram_quantile(0.95, rate(model_prediction_duration_seconds_bucket[5m]))`
   - Guardrail rejections: `mri_guardrail_rejections_total`

## Re-deploying after a new push

```bash
cd ~/neurosight
docker compose -f docker-compose.prod.yml --env-file .env pull
docker compose -f docker-compose.prod.yml --env-file .env up -d
```

## Notes / gotchas

- Two model checkpoints are baked into the backend image under `/model`: `pdscnn_standalone_v2/` (~1.2 MB) and `vit_finetuned/` (~327 MB). The app soft-vote ensembles them, so both directories (checkpoint + `config.json`) are required at runtime.
- Prediction is CPU-only. A request takes roughly 60 s: it loads both models, runs Grad-CAM, and computes SHAP attribution. The Node bridge enforces `PREDICT_TIMEOUT_MS` (default 300000).
- Never commit the real `.env` — `*.env*` is gitignored.