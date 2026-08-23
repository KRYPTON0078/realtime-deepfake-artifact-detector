# Refresh GitHub profile streak badge cache ONLY (keeps S rank + 13k commits).
# Run from repo root: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/update-profile-streak-badge.ps1

$ErrorActionPreference = "Stop"
$stamp = Get-Date -Format "yyyyMMddHHmmss"
$work = Join-Path $env:TEMP "KRYPTON0078-profile-sync"
$repo = "https://github.com/KRYPTON0078/KRYPTON0078.git"

if (Test-Path $work) { Remove-Item -Recurse -Force $work }
git clone $repo $work | Out-Host
Set-Location $work

git config user.name "Magne Dina Neves"
git config user.email "magnedinanevesdina@gmail.com"

$readme = Get-Content "README.md" -Raw
# Only bust cache timestamp — do NOT change include_all_commits/count_private (breaks rank/commits).
$readme = $readme -replace 'v=\d+', "v=$stamp"
$readme = $readme -replace 'date_format=[^&]+&', ''
Set-Content "README.md" $readme -NoNewline

git add README.md
git commit -m "Refresh stats badge cache (restore S rank and streak)."
git push origin main
Write-Host "Done. Hard-refresh https://github.com/KRYPTON0078 (Ctrl+F5). Cache bust v=$stamp"
