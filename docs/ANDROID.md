# Android App (WebView Wrapper)

This project includes a minimal Android wrapper in `android/` that loads the Flask app in a WebView.

## What it does

- Launches a WebView to the backend URL.
- Supports JavaScript and DOM storage for the existing web UI.
- Reuses the upload flow from the Flask app.
- Supports live device camera analysis through browser `getUserMedia` + `/analyze/frame`.
- Draws face-box overlays returned by `/analyze/frame` on the device preview.

## Prerequisites

- Android Studio Iguana or newer
- Android SDK 34
- Flask backend running on a host reachable by your phone/emulator

## Configure backend URL

Default (`android/app/build.gradle`):

```gradle
buildConfigField("String", "BASE_URL", "\"http://10.0.2.2:5000/\"")
```

- `10.0.2.2` works for Android emulator (host machine loopback).
- For a real device on LAN:

```bash
./gradlew assembleDebug -PBACKEND_URL=http://192.168.1.23:5000/
```

Cleartext HTTP is permitted for LAN demos via `network_security_config.xml`. Use HTTPS for any public deployment.

## Run backend for mobile access

```bash
APP_HOST=0.0.0.0 APP_PORT=5000 python app/server.py
```

Ensure your phone/emulator can reach that host and port on the same network.

## Build and run

1. Open `android/` in Android Studio.
2. Let Gradle sync.
3. Run the `app` module on emulator/device.

## Notes

- The "Live Webcam Analysis" panel controls **server-side** `cv2.VideoCapture` (desktop/host camera).
- On phones, use **Device Camera (Mobile/WebView)** which posts frames to `POST /analyze/frame`.
- Capture canvas targets ~640×480 with adaptive polling based on `processing_ms`.
