# Start Here

## Open this project in Cursor

1. In Cursor, click **File -> Open Folder**
2. Select: `D:\DeepfakeDetector`
3. If Cursor asks to **Trust** the workspace, click **Trust**

Do not open `D:\Master_Application` for this project. That is a different folder.

## Run the app

Option A:

```powershell
cd D:\DeepfakeDetector
python app/server.py
```

Option B: double-click `run.bat`

Then open in your browser:

`http://127.0.0.1:5000`

## Git commits

Auto-commit is **disabled** by default (`.cursor/hooks.json` has no active hooks). Commit and push manually when you are ready.

## If Cursor shows a hooks / workspace error

If opening the folder fails because of hooks:

1. Temporarily rename `.cursor/hooks.json` to `.cursor/hooks.json.bak`
2. Re-open `D:\DeepfakeDetector`
3. After the workspace opens, rename it back if you want to re-enable hooks

To re-enable auto-commit, restore the `afterFileEdit` entry in `.cursor/hooks.json` (see `GITHUB_SETUP.md`).

## If the browser page fails to load

- Make sure the terminal shows `Running on http://127.0.0.1:5000`
- Stop any old Flask process using port 5000
- Run again with `python app/server.py`

## If webcam does not work

- Click **Start Camera** on the page
- Allow browser/camera permissions
- Use Chrome or Edge

## Quick test without webcam

Use the **Upload Video** section and analyze a short `.mp4` clip.
