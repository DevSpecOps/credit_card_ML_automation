Write-Host "Building Docker image..." -ForegroundColor Cyan
docker compose build

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Starting container..." -ForegroundColor Cyan
docker compose up -d

Write-Host "Waiting for API to be ready..." -ForegroundColor Cyan
$maxRetries = 10
$retryCount = 0
$ready = $false

while ($retryCount -lt $maxRetries) {
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 2
        if ($response.status -eq "ok") {
            $ready = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 2
        $retryCount++
    }
}

if ($ready) {
    Write-Host "API is UP and running at http://127.0.0.1:8000/health" -ForegroundColor Green
} else {
    Write-Host "API failed to start. Check logs with: docker compose logs" -ForegroundColor Red
}
