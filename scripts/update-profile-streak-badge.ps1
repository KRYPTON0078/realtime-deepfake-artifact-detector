# Refresh GitHub profile streak badge (fixes cached "0" streak bar).
# Run from repo root on your Windows machine (uses YOUR GitHub login, not cursor bot):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/update-profile-streak-badge.ps1

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
$readme = $readme -replace 'v=\d+', "v=$stamp"
$readme = $readme -replace 'include_all_commits=false', 'include_all_commits=true'
$readme = $readme -replace 'count_private=false', 'count_private=true'
$readme = $readme -replace 'cache_seconds=1800', 'cache_seconds=60'
$readme = $readme -replace 'cache_seconds=300', 'cache_seconds=60'
Set-Content "README.md" $readme -NoNewline

git add README.md
git commit -m "Refresh stats badge cache to restore streak display."
git push origin main
Write-Host "Profile README updated (cache bust v=$stamp). Hard-refresh https://github.com/KRYPTON0078"
