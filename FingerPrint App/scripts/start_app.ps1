$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$dataDirectory = Join-Path $projectRoot "data"
$stateFile = Join-Path $dataDirectory "streamlit-process.json"
$stdoutLog = Join-Path $dataDirectory "streamlit-output.log"
$stderrLog = Join-Path $dataDirectory "streamlit-error.log"
$appUrl = "http://127.0.0.1:8501"

function Test-AppPort {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $client.Connect("127.0.0.1", 8501)
        return $true
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

New-Item -ItemType Directory -Force -Path $dataDirectory | Out-Null

if (Test-Path -LiteralPath $stateFile) {
    try {
        $state = Get-Content -Raw -LiteralPath $stateFile | ConvertFrom-Json
        $existing = Get-Process -Id ([int]$state.pid) -ErrorAction Stop
        $actualStart = $existing.StartTime.ToUniversalTime().Ticks
        if ($actualStart -eq [long]$state.startTimeUtcTicks) {
            Write-Host "Fingerprint Attendance is already running at $appUrl" -ForegroundColor Green
            Start-Process $appUrl
            exit 0
        }
    }
    catch {
        Remove-Item -LiteralPath $stateFile -Force -ErrorAction SilentlyContinue
    }
}

if (Test-AppPort) {
    Write-Host "An app is already running on $appUrl. No duplicate server was started." -ForegroundColor Yellow
    Write-Host "If it was started manually, stop it with Ctrl+C in its original terminal."
    Start-Process $appUrl
    exit 0
}

$pythonCommand = Get-Command python -ErrorAction Stop
$appPath = Join-Path $projectRoot "app.py"
$arguments = @(
    "-m", "streamlit", "run", "`"$appPath`"",
    "--server.port", "8501",
    "--server.address", "127.0.0.1",
    "--server.headless", "true",
    "--browser.gatherUsageStats", "false"
)
$process = Start-Process `
    -FilePath $pythonCommand.Source `
    -ArgumentList $arguments `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

$process.Refresh()
@{
    pid = $process.Id
    startTimeUtcTicks = $process.StartTime.ToUniversalTime().Ticks
} | ConvertTo-Json | Set-Content -LiteralPath $stateFile -Encoding UTF8

$ready = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Milliseconds 350
    if ($process.HasExited) { break }
    try {
        if (-not (Test-AppPort)) { throw "not ready" }
        $ready = $true
        break
    }
    catch {}
}

if (-not $ready) {
    Remove-Item -LiteralPath $stateFile -Force -ErrorAction SilentlyContinue
    throw "The app did not start. Check data\streamlit-error.log for details."
}

Write-Host "Fingerprint Attendance started at $appUrl" -ForegroundColor Green
Start-Process $appUrl
