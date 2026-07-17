# GitHub Setup

Local git is initialized and the initial commit is ready.

## Create the GitHub repository

1. Go to GitHub and create a new empty repository:
   - Name: `realtime-deepfake-artifact-detector`
   - Do not initialize with README (this project already has one)

2. Push from PowerShell:

```powershell
cd D:\DeepfakeDetector
$env:GIT_SSL_NO_VERIFY = "true"
git remote remove origin 2>$null
git remote add origin https://github.com/KRYPTON0078/realtime-deepfake-artifact-detector.git
git push -u origin main
```

## Auto-commit hooks (disabled by default)

Scripts are kept for optional use:

- `.cursor/hooks/auto-commit.ps1`
- `.cursor/hooks/auto-commit-worker.ps1`

Auto-commit is **off** in `.cursor/hooks.json` (`"hooks": {}`). Commit and push manually, or re-enable by adding an `afterFileEdit` hook that runs `auto-commit.ps1` (60-second debounce, then commit and push to `main`).

Logs when enabled: `.cursor/hooks/auto-commit.log`
