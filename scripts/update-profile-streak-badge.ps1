# One-time helper: refresh the streak badge cache on your GitHub profile README.
# Run from a clone of https://github.com/KRYPTON0078/KRYPTON0078

$stamp = Get-Date -Format "yyyyMMddHHmmss"
$readme = Join-Path $PSScriptRoot "..\..\KRYPTON0078\README.md"
if (-not (Test-Path $readme)) {
    Write-Host "Clone KRYPTON0078/KRYPTON0078 next to this repo, then rerun."
    exit 1
}

$content = Get-Content $readme -Raw
$content = $content -replace 'v=\d+', "v=$stamp"
$content = $content -replace 'include_all_commits=false', 'include_all_commits=true'
$content = $content -replace 'count_private=false', 'count_private=true'
$content = $content -replace 'cache_seconds=1800', 'cache_seconds=60'
Set-Content $readme $content -NoNewline

Push-Location (Split-Path $readme)
git add README.md
git commit -m "Refresh stats badge cache to restore streak display."
git push origin main
Pop-Location
Write-Host "Profile README pushed with cache bust v=$stamp"
