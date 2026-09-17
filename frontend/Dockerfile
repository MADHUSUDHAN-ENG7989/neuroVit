# Frontend Dockerfile (multi-stage) for NeuroSight AI
# Build the React app with Vite and serve using nginx

FROM node:20-alpine AS builder
WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci --omit=dev

# Copy source code and build
COPY . ./
RUN npm run build

# ---- Runtime ----
FROM nginx:stable-alpine AS runtime
COPY --from=builder /app/dist /usr/share/nginx/html
# Expose the default nginx port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
