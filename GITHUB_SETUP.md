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

## Auto-commit hooks

Project hooks are configured in:

- `.cursor/hooks.json`
- `.cursor/hooks/auto-commit.ps1`
- `.cursor/hooks/auto-commit-worker.ps1`

Open `D:\DeepfakeDetector` as the Cursor workspace root so Agent edits auto-commit and push to `main` after a 60-second debounce window.

Logs: `.cursor/hooks/auto-commit.log`
