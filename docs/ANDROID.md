# Android App (WebView Wrapper)

This project includes a minimal Android wrapper in `android/` that loads the Flask app in a WebView.

## What it does

- Launches a WebView to the backend URL.
- Supports JavaScript and DOM storage for the existing web UI.
- Reuses the upload flow from the Flask app.
- Supports live device camera analysis through browser `getUserMedia` + `/analyze/frame`.

## Prerequisites

- Android Studio Iguana or newer
- Android SDK 34
- Flask backend running on a host reachable by your phone/emulator

## Configure backend URL

`android/app/build.gradle` defines:

```gradle
buildConfigField("String", "BASE_URL", "\"http://10.0.2.2:5000/\"")
```

- `10.0.2.2` works for Android emulator (host machine loopback).
- For a real device, replace with your computer's LAN IP, for example:
  `http://192.168.1.23:5000/`

## Run backend for mobile access

From project root:

```powershell
$env:APP_HOST = "0.0.0.0"
$env:APP_PORT = "5000"
python app/server.py
```

Ensure your phone/emulator can reach that host and port on the same network.

## Build and run

1. Open `android/` in Android Studio.
2. Let Gradle sync.
3. Run the `app` module on emulator/device.

## Current limitation

- The "Live Webcam Analysis" panel still controls **server-side** camera capture (`cv2.VideoCapture`).
- For phone camera analysis, use the "Device Camera (Mobile/WebView)" panel, which samples camera frames and sends them to `POST /analyze/frame`.
