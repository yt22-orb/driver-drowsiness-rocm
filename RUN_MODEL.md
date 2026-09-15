# Run the custom driver-drowsiness model

This guide runs the repository-owned CNN locally in Chrome or Edge. Inference uses WebGPU when
available and falls back to WebAssembly on the CPU.

> **Safety:** This is an experimental thesis prototype, not a certified automotive safety system.
> Use it only while safely seated and stationary. Never operate or test it while driving.

## 1. Produce the model assets

There is no compatible third-party checkpoint or detector to download. Build the model assets from
this repository's training pipeline:

```bash
make rocm-build
make rocm-check
make audit
make train
make evaluate
make export
```

The export step creates:

```text
web/public/models/drowsiness-cnn-v1.onnx
web/public/models/model-metadata.json
```

For the two-model selector, the app also expects the architecture-specific custom metadata and the
legacy GitHub Release assets:

```text
web/public/models/drowsiness-cnn-v1.metadata.json
web/public/models/drowsiness-mobilenet-v3-small.onnx
web/public/models/mobilenet-v3-small.metadata.json
```

Run `make legacy-model` to download the two legacy files from your `v0.1.0-model` GitHub Release.

The metadata must have schema version 2, architecture `drowsiness_cnn_v1`, and an
`externalCheckpoint` value of `false`. The browser rejects legacy or unrelated metadata.

## 2. Start the app

```bash
npm --prefix web ci
npm --prefix web run dev
```

Open <http://127.0.0.1:5173>, wait for **Ready**, click **Start camera**, and allow camera access.
Use the **Model** selector to switch between the current custom CNN and the released MobileNetV3.
Switching models stops an active camera session so that the selected model's preprocessing and
decision threshold can be initialized safely.

## Required framing

The CNN was trained on face-crop images and the app intentionally has no separate face or landmark
model. Keep exactly one face centered and filling the dashed square shown over the camera image.
For **Test an image**, choose a JPEG or PNG with one similarly centered face.

The custom CNN takes the centered square whose side is 70% of the source's shorter dimension,
resizes it to 224 × 224, and scales RGB values to `[-1, 1]`. The legacy MobileNetV3 uses its original
MediaPipe face detector and ImageNet normalization. Both supply a float32 tensor shaped
`[1, 3, 224, 224]`. A badly framed image may still produce a confident but meaningless score.

## Display behavior

- **Drowsiness score** is softmax probability for label 0, `Drowsy`.
- **No drowsiness detected** means the smoothed score is below the calibrated threshold.
- **Checking sustained signal** means a high score has not yet filled the time window.
- **Possible drowsiness detected** means the sustained score triggered the warning.

The camera alarm uses the timing values exported in `model-metadata.json`. Uploaded images display a
single score and do not sound the alarm.

## Troubleshooting

- Camera access requires `127.0.0.1`, `localhost`, or HTTPS.
- Close applications that have exclusive camera control.
- Use even front lighting and keep the full face inside the crop guide.
- If WebGPU is unavailable, the app automatically uses the slower CPU runtime.
- If model initialization fails, confirm both generated files exist in `web/public/models/` and
  rerun `make export`; opening `index.html` directly is unsupported.

The dataset does not include driver identities and its card declares no license. Results must not be
interpreted as proof of generalization or a grant to redistribute the source data.
