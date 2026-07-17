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

Auto-commit is **enabled**. When the Cursor Agent edits files, changes are committed and pushed to `main` automatically after a 60-second debounce window.

Logs: `.cursor/hooks/auto-commit.log`

## If Cursor shows a hooks / workspace error

The auto-commit hook is optional. If opening the folder fails because of hooks:

1. Temporarily rename `.cursor/hooks.json` to `.cursor/hooks.json.bak`
2. Re-open `D:\DeepfakeDetector`
3. After the workspace opens, rename it back if you want auto-commit again

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
