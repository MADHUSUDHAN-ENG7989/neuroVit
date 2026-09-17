# Backend Dockerfile for NeuroSight AI
# Uses Node 20 (alpine) and installs Python for the prediction script

FROM node:20-slim AS base

# Install Python & pip (needed for model inference)
RUN apt-get update && apt-get install -y python3 python3-pip && ln -sf python3 /usr/bin/python
# ln -sf python3 /usr/bin/python (removed, handled above)

WORKDIR /app

# Copy package files and install Node dependencies
COPY package*.json ./
RUN npm ci --omit=dev

# Copy backend source code and model files
COPY . ./

# Install Python dependencies (if any)
# Expect a requirements.txt in backend (create if missing)
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir --break-system-packages -r requirements.txt; fi

# Expose backend port
EXPOSE 5000

# Runtime command
CMD ["node", "server.js"]
