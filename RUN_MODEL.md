# Run the trained driver-drowsiness model

This guide runs the released model locally in Chrome or Edge. It does **not** require the training
dataset, Python, PyTorch, ROCm, CUDA, or an AMD GPU. Inference uses WebGPU when available and falls
back to WebAssembly on the CPU.

> **Safety:** This is an experimental thesis prototype, not a certified automotive safety system.
> Try it only while safely seated and stationary. Never operate or test it while driving.

## Requirements

- Git
- Node.js 22 or newer and npm
- Chrome or Edge with camera permission
- An internet connection for the initial model and browser-runtime downloads

Check Node.js before continuing:

```bash
node --version
npm --version
```

## 1. Download the code

```bash
git clone https://github.com/yt22-orb/driver-drowsiness-rocm.git
cd driver-drowsiness-rocm
git switch agent/runnable-model-release
```

After the pull request is merged, the final `git switch` command is unnecessary because the files
will be on `main`.

## 2. Download the released browser model

On Linux, macOS, or Git Bash:

```bash
mkdir -p web/public/models
curl -fL \
  -o web/public/models/drowsiness-mobilenet-v3-small.onnx \
  https://github.com/yt22-orb/driver-drowsiness-rocm/releases/download/v0.1.0-model/drowsiness-mobilenet-v3-small.onnx
curl -fL \
  -o web/public/models/model-metadata.json \
  https://github.com/yt22-orb/driver-drowsiness-rocm/releases/download/v0.1.0-model/model-metadata.json
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force web/public/models | Out-Null
Invoke-WebRequest `
  -Uri "https://github.com/yt22-orb/driver-drowsiness-rocm/releases/download/v0.1.0-model/drowsiness-mobilenet-v3-small.onnx" `
  -OutFile "web/public/models/drowsiness-mobilenet-v3-small.onnx"
Invoke-WebRequest `
  -Uri "https://github.com/yt22-orb/driver-drowsiness-rocm/releases/download/v0.1.0-model/model-metadata.json" `
  -OutFile "web/public/models/model-metadata.json"
```

The release also includes the original PyTorch checkpoint for research and reproducibility. The
browser needs only the ONNX file and `model-metadata.json`.

## 3. Start the camera app

```bash
npm --prefix web ci
npm --prefix web run dev
```

Open <http://127.0.0.1:5173>. Wait for **Ready**, click **Start camera**, and allow camera access.
You can also click **Test an image** to classify a JPEG or PNG containing a visible face.

Keep the terminal running while using the app. Stop the server with `Ctrl+C`.

## What the display means

- **Drowsiness score** is the model's probability for the drowsy class.
- **No drowsiness detected** means the smoothed score is below the calibrated threshold.
- **Checking sustained signal** means the score is high but the required time window is incomplete.
- **Possible drowsiness detected** means the high score persisted long enough to trigger the alarm.
- **No face detected** means the face detector could not produce a crop for the classifier.

The alarm uses a two-second score window and requires at least 15 valid face frames. Uploaded still
images display a score but do not sound the sustained-warning alarm.

## Camera troubleshooting

- Use `127.0.0.1` or `localhost`; browsers treat these as secure camera contexts.
- If permission was blocked, use the lock/camera icon beside the address, set Camera to **Allow**,
  and reload.
- Close another application if it has exclusive control of the camera.
- Improve front lighting and keep the full face visible.
- If WebGPU is unavailable, the app automatically uses the slower WASM CPU runtime.

## Recorded model results

The epoch-15 MobileNetV3-Small checkpoint scored the following on the untouched official test split
of 8,359 images:

| Metric | Result |
| --- | ---: |
| Accuracy | 99.9880% |
| Balanced accuracy | 99.9888% |
| Drowsy precision | 100.0000% |
| Drowsy recall | 99.9777% |
| Drowsy F1 | 99.9888% |
| Matthews correlation coefficient | 99.9760% |
| ROC-AUC | 100.0000% |
| Confusion matrix `[Drowsy, Non Drowsy]` | `[[4479, 1], [0, 3879]]` |

These results are unusually high and should not be interpreted as proof of real-world reliability.
The dataset does not supply driver identities, so driver-disjoint generalization could not be
measured. The dataset card also declares no license; review the source dataset's terms before using
the released derived weights beyond research or evaluation.
