# Webcam Hand Tracking Prototype

A usable local prototype that tracks up to two hands from a normal webcam and maps them to simple digital 3D hand skeletons.

## What it does

- Requests webcam access.
- Loads Google's MediaPipe Hand Landmarker task.
- Detects up to two hands.
- Renders 21 hand landmarks and bone connections in Three.js.
- Shows FPS, handedness, confidence, and basic gesture labels:
  - Pinch
  - Open palm
  - Fist / grab
  - Point
  - Neutral

## Requirements

- Node.js 18+
- A modern Chromium, Edge, Safari, or Firefox browser
- A webcam
- Internet access on first run to download MediaPipe WASM/model assets

## Run locally

```bash
cd hand-tracking-prototype
npm install
npm run dev
```

Then open the local URL shown by Vite, usually:

```text
http://localhost:5173
```

Camera access normally works on `localhost` or HTTPS. It may not work from a plain `file://` URL.

## Notes

This is intentionally a skeleton prototype, not a realistic skinned hand mesh. It proves the interaction pipeline first:

```text
webcam → landmarks → smoothing → digital hand model → gesture readout
```

The next upgrade would be to map the 21 landmarks to a rigged hand mesh and add a gameplay input state machine.
