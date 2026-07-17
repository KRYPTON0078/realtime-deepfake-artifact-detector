param(
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [int]$DebounceSeconds = 60
)

$ErrorActionPreference = "SilentlyContinue"

$HookDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$StateFile = Join-Path $HookDir ".debounce-state.json"
$LockFile = Join-Path $HookDir ".worker.lock"
$LogFile = Join-Path $HookDir "auto-commit.log"

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $LogFile -Value $line
}

function Get-LastEditTime {
    if (-not (Test-Path $StateFile)) { return $null }
    try {
        $state = Get-Content $StateFile -Raw | ConvertFrom-Json
        return [datetime]::Parse($state.lastEdit)
    } catch {
        return $null
    }
}

try {
    Set-Location $RepoRoot
    $env:GIT_SSL_NO_VERIFY = "true"
    Write-Log "Worker started for repo: $RepoRoot"

    while ($true) {
        Start-Sleep -Seconds 5
        $lastEdit = Get-LastEditTime
        if (-not $lastEdit) { continue }

        $idleSeconds = ((Get-Date) - $lastEdit).TotalSeconds
        if ($idleSeconds -lt $DebounceSeconds) { continue }

        $status = git status --porcelain 2>&1
        if (-not $status) {
            Write-Log "No changes to commit"
            break
        }

        git add -A 2>&1 | Out-Null
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $commitMessage = "auto: agent update ($timestamp)"
        git commit -m $commitMessage 2>&1 | ForEach-Object { Write-Log $_ }

        if ($LASTEXITCODE -ne 0) {
            Write-Log "Commit skipped or failed (exit $LASTEXITCODE)"
            break
        }

        $pushOutput = git push origin HEAD:main 2>&1
        foreach ($line in $pushOutput) { Write-Log $line }

        if ($LASTEXITCODE -eq 0) {
            Write-Log "Push succeeded to origin/main"
        } else {
            Write-Log "Push failed (exit $LASTEXITCODE). Changes remain committed locally."
        }
        break
    }
} catch {
    Write-Log "Worker error: $($_.Exception.Message)"
} finally {
    if (Test-Path $LockFile) { Remove-Item $LockFile -Force }
    Write-Log "Worker finished"
}
