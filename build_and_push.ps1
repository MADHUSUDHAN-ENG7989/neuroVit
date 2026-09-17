# PowerShell Script to Build and Push Docker Images to Docker Hub

param (
    [string]$DockerUser = "",
    [string]$Tag = "latest"
)

if ([string]::IsNullOrWhiteSpace($DockerUser)) {
    $DockerUser = Read-Host -Prompt "Enter your Docker Hub Username"
}

if ([string]::IsNullOrWhiteSpace($DockerUser)) {
    Write-Error "Docker Hub Username is required."
    exit 1
}

$BackendImage = "$DockerUser/neurosight-backend:$Tag"
$FrontendImage = "$DockerUser/neurosight-frontend:$Tag"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " Building & Pushing Docker Images to Docker Hub" -ForegroundColor Cyan
Write-Host " Docker Username : $DockerUser" -ForegroundColor Yellow
Write-Host " Tag             : $Tag" -ForegroundColor Yellow
Write-Host " Backend Image   : $BackendImage" -ForegroundColor Yellow
Write-Host " Frontend Image  : $FrontendImage" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Cyan

# Check Docker Status
Write-Host "`n[1/5] Checking Docker connection..." -ForegroundColor Green
docker info > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Initial connection check returned code $LASTEXITCODE. Setting API negotiation variable..."
    $env:DOCKER_API_VERSION = "1.47"
    docker info > $null 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Docker Desktop Engine is not currently responding (HTTP 500 / pipe disconnected)."
        Write-Warning "Please restart Docker Desktop from your Windows Start Menu as Administrator."
    }
}

# Login check
Write-Host "`n[2/5] Logging into Docker Hub (if prompted)..." -ForegroundColor Green
docker login

# Build Backend Image
Write-Host "`n[3/5] Building Backend Docker Image ($BackendImage)..." -ForegroundColor Green
docker build -t $BackendImage ./backend
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to build backend image."
    exit 1
}

# Build Frontend Image
Write-Host "`n[4/5] Building Frontend Docker Image ($FrontendImage)..." -ForegroundColor Green
docker build -t $FrontendImage ./frontend
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to build frontend image."
    exit 1
}

# Push Images
Write-Host "`n[5/5] Pushing Docker Images to Docker Hub..." -ForegroundColor Green
Write-Host "Pushing $BackendImage..." -ForegroundColor Yellow
docker push $BackendImage

Write-Host "Pushing $FrontendImage..." -ForegroundColor Yellow
docker push $FrontendImage

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host " Successfully built and pushed images!" -ForegroundColor Green
Write-Host " Backend:  https://hub.docker.com/r/$DockerUser/neurosight-backend" -ForegroundColor Cyan
Write-Host " Frontend: https://hub.docker.com/r/$DockerUser/neurosight-frontend" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
