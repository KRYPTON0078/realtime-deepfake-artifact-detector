# Cursor afterFileEdit hook: debounce Agent edits, then commit and push.
$ErrorActionPreference = "SilentlyContinue"

$HookDir = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $HookDir "..\..")).Path
$StateFile = Join-Path $HookDir ".debounce-state.json"
$LockFile = Join-Path $HookDir ".worker.lock"
$LogFile = Join-Path $HookDir "auto-commit.log"
$WorkerScript = Join-Path $HookDir "auto-commit-worker.ps1"
$DebounceSeconds = 60

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $LogFile -Value $line
}

try {
    $stdin = [Console]::In.ReadToEnd()
    $editedFile = $null
    if ($stdin) {
        try {
            $payload = $stdin | ConvertFrom-Json
            if ($payload.file_path) { $editedFile = [string]$payload.file_path }
            elseif ($payload.path) { $editedFile = [string]$payload.path }
            elseif ($payload.filePath) { $editedFile = [string]$payload.filePath }
        } catch {
            Write-Log "Could not parse hook JSON: $($_.Exception.Message)"
        }
    }

    $state = [ordered]@{
        lastEdit = (Get-Date).ToString("o")
        files = @()
    }
    if (Test-Path $StateFile) {
        try {
            $existing = Get-Content $StateFile -Raw | ConvertFrom-Json
            if ($existing.files) { $state.files = @($existing.files) }
        } catch {}
    }
    if ($editedFile -and ($state.files -notcontains $editedFile)) {
        $state.files += $editedFile
    }
    ($state | ConvertTo-Json -Depth 4) | Set-Content -Path $StateFile -Encoding UTF8

    $workerRunning = $false
    if (Test-Path $LockFile) {
        $workerPid = Get-Content $LockFile -ErrorAction SilentlyContinue
        if ($workerPid -and (Get-Process -Id $workerPid -ErrorAction SilentlyContinue)) {
            $workerRunning = $true
        }
    }

    if (-not $workerRunning) {
        $proc = Start-Process -FilePath "powershell.exe" -PassThru -WindowStyle Hidden -ArgumentList @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", "`"$WorkerScript`"",
            "-RepoRoot", "`"$RepoRoot`"",
            "-DebounceSeconds", "$DebounceSeconds"
        )
        Set-Content -Path $LockFile -Value $proc.Id -Encoding ASCII
        Write-Log "Started debounce worker (PID $($proc.Id))"
    } else {
        Write-Log "Debounce worker already running; updated state"
    }
} catch {
    Write-Log "Hook error: $($_.Exception.Message)"
}

exit 0
