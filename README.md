# Driver Drowsiness ROCm

An experimental computer-vision system that fine-tunes MobileNetV3-Small on the
[Driver Drowsiness Dataset](https://huggingface.co/datasets/akahana/Driver-Drowsiness-Dataset),
exports the classifier to ONNX, and runs webcam or image inference locally in a browser.

The training environment targets an AMD Radeon RX 7800 XT with ROCm 7.2.1 and PyTorch 2.9.1.
No NVIDIA hardware, CUDA build, hosted inference API, or cloud camera processing is required.

> [!WARNING]
> This is an experimental thesis prototype, not a certified automotive safety system. Do not
> rely on it to decide whether it is safe to drive. The source dataset has no subject identifiers,
> so the supplied split does not establish generalization to unseen drivers.

## Repository and licensing boundary

This public repository contains original source code and documentation only. It intentionally has
no software license, which means no reuse permission is granted by default. The source dataset does
not declare a license on its Hugging Face card. Dataset files, copied sample images, trained
checkpoints, ONNX weights, and other derived model artifacts are therefore ignored and must not be
published until their rights are resolved.

## Architecture

```text
Hugging Face images (227x227 face crops)
        |
        +-- exact-duplicate audit manifest
        +-- stratified train/validation split (seed 42)
        |
MobileNetV3-Small, ImageNet initialization, 224x224 RGB
        |
        +-- validation F2 selects checkpoint and drowsy threshold
        +-- official test split is evaluated once
        |
FP32 ONNX model + JSON inference contract
        |
MediaPipe face crop -> ONNX Runtime Web -> 2-second warning window
```

The immutable dataset contract is:

| Property | Value |
| --- | --- |
| Dataset | `akahana/Driver-Drowsiness-Dataset` |
| Revision | `1770cfcacac05ff4ae280479ed610c3fcc6b4b7c` |
| Official train rows | 33,434 |
| Official test rows | 8,359 |
| Label `0` | `Drowsy` (the positive class) |
| Label `1` | `Non Drowsy` |

The browser model contract is fixed to float32 `images` shaped `[1, 3, 224, 224]`, normalized with
ImageNet mean/std, and float32 `logits` shaped `[1, 2]` in the label order above.

## RX 7800 XT prerequisites

Use one of AMD's supported host combinations, such as Ubuntu 22.04.5 Desktop HWE with kernel 6.8,
or Ubuntu 24.04.4 Desktop HWE with kernel 6.17. Install the matching Radeon Software for Linux with
ROCm and Docker Engine. AMD lists the RX 7800 XT, ROCm 7.2.1, and PyTorch 2.9.1 as an officially
supported production combination:

- [ROCm Radeon Linux support matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html)
- [Install PyTorch for ROCm on Radeon](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html)

Confirm the host exposes both device nodes before building:

```bash
ls -l /dev/kfd /dev/dri
docker --version
docker compose version
```

## Run the complete rig workflow

Clone and enter the repository, then run each stage separately so a failed stage is obvious:

```bash
git clone https://github.com/yt22-orb/driver-drowsiness-rocm.git
cd driver-drowsiness-rocm
git fetch origin Ken/initial-mvp
git switch --track origin/Ken/initial-mvp

make rocm-build
make rocm-check
make audit
make train
make evaluate
make export
```

`make audit` downloads/caches the dataset, validates every decoded image, counts dimensions and
classes, and writes `artifacts/audit/dedup-manifest.json`. Training preserves official test copies,
removes exact cross-split train copies, and keeps only the first exact duplicate within training.
Perceptual-hash collisions are reported as suspects but are not removed automatically.

Expected outputs:

| Artifact | Path |
| --- | --- |
| Audit report | `artifacts/audit/dataset-audit.json` |
| Dedup manifest | `artifacts/audit/dedup-manifest.json` |
| Resolved config | `artifacts/runs/mobilenet-v3-small/resolved-config.json` |
| Epoch history | `artifacts/runs/mobilenet-v3-small/history.jsonl` |
| Best checkpoint | `artifacts/runs/mobilenet-v3-small/best.pt` |
| Test metrics | `artifacts/evaluation/test-metrics.json` |
| ONNX model | `artifacts/export/drowsiness-mobilenet-v3-small.onnx` |
| Inference contract | `artifacts/export/model-metadata.json` |
| Browser copies | `web/public/models/` |

Training defaults live in [`configs/mvp.yaml`](configs/mvp.yaml). The RX 7800 XT default batch size
is 128. If it runs out of memory, rerun training with `--batch-size 64` inside the Compose command
shown by `make train`, or change only `training.batch_size` in the YAML and record the change.

## Run browser inference

The export command copies the ignored ONNX model and metadata into `web/public/models/`. After that:

```bash
make web-install
make web-dev
```

Open <http://127.0.0.1:5173>. `localhost` is a secure browser context, so camera permission works.
The app uses WebGPU when available and ONNX Runtime WASM otherwise. MediaPipe finds the largest
face, adds 15% crop padding, and the classifier runs at a target 10 Hz. An alarm starts only after a
sustained, smoothed drowsy score; single uploaded images show a score but never sound the alarm.

All frames and scores stay in the browser. The MediaPipe and ONNX Runtime Web runtime assets are
downloaded from their official CDNs when the page initializes, so the current MVP is not fully
offline on first use.

## CPU development on macOS or Linux

The project requires Python 3.12 or 3.13. CPU development does not download the full dataset:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
make install
make lint
make test
make smoke
make metadata-audit

make web-install
make web-build
```

The smoke command performs a real forward/backward optimizer step using synthetic images, exports
a temporary random-weight ONNX model, checks it with `onnx.checker`, and requires PyTorch/ONNX
predictions to match with maximum absolute logit difference below `1e-4`. That smoke model is not
a usable drowsiness classifier.

## Training and evaluation behavior

- MobileNetV3-Small with ImageNet pretrained weights and a new two-logit head.
- Mild horizontal flip, brightness/contrast/saturation jitter, translation, and ±5° rotation.
- Inverse-frequency cross-entropy weights.
- AdamW, learning rate `3e-4`, weight decay `1e-4`, cosine schedule, and FP16 ROCm AMP.
- Maximum 15 epochs and early stopping after four epochs without improved validation drowsy F2.
- The decision threshold is selected only from validation predictions with F2, which weights missed
  drowsy samples more heavily than false alarms.
- The untouched official test split uses the saved checkpoint and frozen validation threshold.
- Reported metrics: accuracy, balanced accuracy, macro F1, drowsy precision/recall/F2, ROC-AUC,
  class counts, and the confusion matrix in `[Drowsy, Non Drowsy]` order.

## Troubleshooting

### `/dev/kfd` is missing

The ROCm kernel driver is not loaded or the host installation/GPU is unsupported. Do not change the
container to fake success. Recheck AMD's host matrix, Radeon driver installation, user membership in
`video`/`render`, and reboot after driver changes.

### `torch.cuda.is_available()` is false

PyTorch intentionally retains the `torch.cuda` API name on ROCm. `make rocm-check` also verifies
`torch.version.hip` and requires a detected device name containing `7800 XT`. If HIP is empty, the
wrong CPU/CUDA wheel replaced AMD's container package; rebuild the pinned image without adding a
separate PyTorch index.

### GPU out of memory

Reduce batch size from 128 to 64, then 32. Record the final batch size and do not change input size,
architecture, label mapping, or test split as an OOM workaround.

### Dataset load or decode fails

Confirm at least 6 GB free for the 2.76 GB download plus decoded/cache overhead. Delete no shared
Hugging Face cache blindly. Save the full exception, dataset revision, failing split/index, and rerun
the metadata audit before changing dependencies.

### ONNX parity fails

Do not publish the exported weights. Preserve the checkpoint, record the maximum difference and
mismatched samples, then verify preprocessing, eval mode, opset 18, output order, and ONNX Runtime
version. The exporter deliberately uses a fixed batch and conservative legacy Torch exporter.

### The browser says the model is missing

This is expected before `make export`. Confirm both ignored files exist in `web/public/models/` and
serve the Vite app from `web/`; opening `index.html` directly will not work.

### Camera access fails

Use `http://127.0.0.1`, `http://localhost`, or HTTPS; camera access is blocked on ordinary insecure
origins. Check browser site permissions and close other apps that exclusively own the camera.

### WebGPU is unavailable

The app falls back to WASM CPU and displays the selected runtime. Update the browser and GPU driver
before treating lower WASM performance as a model defect.

### The alarm never sounds or will not clear

Click **Start camera** once to unlock Web Audio. Warning requires at least 15 valid face frames over
roughly two seconds. It clears after a full second below `threshold - 0.15`, or after face loss. Check
the mute button and exported decision values in `model-metadata.json` before changing the UI.

## Continue on the ROCm rig with Codex

Codex on the Ubuntu rig must read this entire README before editing or running commands. Treat the
configuration, dataset revision, labels, official test boundary, ONNX I/O, and publication boundary
as fixed decisions. Do not commit the dataset, weights, copied samples, or generated artifacts.

Handoff state:

| Item | Value |
| --- | --- |
| Repository | `yt22-orb/driver-drowsiness-rocm` |
| Branch | `Ken/initial-mvp` |
| Pull request | [#1 — Build ROCm driver drowsiness MVP](https://github.com/yt22-orb/driver-drowsiness-rocm/pull/1) |
| Implementation handoff commit | `d42763f` |

Verification boundary:

| Verified on the Mac | Must be verified on the RX 7800 XT rig |
| --- | --- |
| Python formatting and lint | `/dev/kfd` and RX 7800 XT detection |
| Unit tests with synthetic data | Full 2.76 GB dataset audit/download |
| CPU forward/backward smoke step | ROCm FP16 mixed-precision training |
| Random-weight ONNX graph/parity | Complete model convergence and checkpoint selection |
| TypeScript typecheck and production build | Final official-test metrics |
| CI-equivalent Python/browser commands | Trained ONNX browser score and alarm behavior |

On the rig, copy [`scripts/rig-results-template.md`](scripts/rig-results-template.md) to a dated file,
fill it with exact commands and outputs, and update this README's dated results section. Never invent
metrics. If a step fails, preserve the error and diagnose it before moving to the next stage.

### Rig acceptance checklist

- [ ] `make rocm-check` reports HIP and the RX 7800 XT.
- [ ] Full audit completes with no unexplained corrupt images.
- [ ] Dedup manifest revision matches the pinned dataset commit.
- [ ] Training produces finite losses and a selected checkpoint.
- [ ] Evaluation writes every documented metric using the frozen threshold.
- [ ] ONNX parity passes on 100 official test images.
- [ ] Browser runs through WebGPU and separately through forced WASM fallback.
- [ ] Webcam no-face, one-face, sustained-warning, clearing, and mute states work.
- [ ] Image upload accepts JPEG/PNG, rejects other types, and never sounds the alarm.
- [ ] Final artifact hashes and observed performance are recorded.

### Dated rig results

No RX 7800 XT run has been performed yet. Add results here only after executing them on the target
Ubuntu machine.
