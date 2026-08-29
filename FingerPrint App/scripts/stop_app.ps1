$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$stateFile = Join-Path $projectRoot "data\streamlit-process.json"
$stopped = 0

# Port 8501 is reserved for this project by .streamlit/config.toml. Resolving
# the listener also lets this shortcut stop an app that was started manually.
try {
    $listeners = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction Stop
}
catch {
    $listeners = @()
}

$processIds = @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)
if (Test-Path -LiteralPath $stateFile) {
    try {
        $state = Get-Content -Raw -LiteralPath $stateFile | ConvertFrom-Json
        $processIds += [int]$state.pid
        $processIds = @($processIds | Select-Object -Unique)
    }
    catch {}
}
foreach ($processId in $processIds) {
    try {
        $process = Get-Process -Id ([int]$processId) -ErrorAction Stop
        if ($process.ProcessName -notmatch "^(python|pythonw)$") {
            Write-Host "Port 8501 belongs to $($process.ProcessName), so it was not stopped." -ForegroundColor Red
            continue
        }
        Stop-Process -Id $process.Id -Force
        $stopped++
    }
    catch {
        # The process may have exited between listener discovery and stopping.
    }
}

Remove-Item -LiteralPath $stateFile -Force -ErrorAction SilentlyContinue
if ($stopped -gt 0) {
    Write-Host "Fingerprint Attendance stopped." -ForegroundColor Green
}
else {
    Write-Host "Fingerprint Attendance is already stopped." -ForegroundColor Yellow
}
