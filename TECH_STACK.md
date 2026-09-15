# Technology Stack

This project trains, evaluates, exports, and runs a custom driver-drowsiness image classifier. The
primary model is a repository-defined convolutional neural network (CNN) trained from random
initialization. An explicitly selected legacy comparison path runs the earlier MobileNetV3 release
with its original MediaPipe face detector.

## Stack at a glance

| Area | Technology | Version | Purpose |
| --- | --- | --- | --- |
| Language | Python | `>=3.12,<3.14` | Model training, evaluation, auditing, and export |
| ML framework | PyTorch | `2.9.1` | Defines and trains the custom CNN |
| Vision utilities | TorchVision | `>=0.24,<0.25` | Image resizing, normalization, and training augmentation |
| GPU platform | AMD ROCm | `7.2.1` | GPU acceleration on the target AMD Radeon RX 7800 XT |
| Dataset library | Hugging Face Datasets | `>=3.2,<5` | Downloads, validates, splits, and caches the image dataset |
| Image library | Pillow | `>=11,<13` | Decodes images and converts them to RGB |
| Numerical computing | NumPy | `>=1.26,<2` | Array processing and PyTorch/ONNX output comparison |
| Metrics | scikit-learn | `>=1.6,<2` | Classification metrics and F2 threshold selection |
| Model format | ONNX | `>=1.17,<2` | Portable model export and graph validation |
| Python inference | ONNX Runtime | `>=1.20,<2` | Checks exported-model parity on the CPU |
| Browser inference | ONNX Runtime Web | `1.27.0` | Runs the ONNX model with WebGPU or WebAssembly |
| Legacy face crop | MediaPipe Tasks Vision | `1.0.1` | Detects faces only for the legacy MobileNet option |
| Configuration | PyYAML | `>=6.0,<7` | Loads the training and export configuration |
| Progress display | tqdm | `>=4.67,<5` | Shows dataset-audit progress |
| Web language | TypeScript | `7.0.2` | Implements the browser inference application |
| Web build tool | Vite | `8.2.1` | Development server and production web build |
| Containerization | Docker and Docker Compose | Host-provided | Reproduces the ROCm training environment |
| Python packaging | Hatchling | `>=1.27` | Builds the Python package |

Versions above come from [`pyproject.toml`](pyproject.toml), [`web/package.json`](web/package.json),
and [`Dockerfile.rocm`](Dockerfile.rocm). A range indicates the supported dependency constraint;
the installed or locked environment may select a specific version within that range.

## Model and training stack

The `drowsiness_cnn_v1` architecture is implemented directly with `torch.nn` in
[`src/drowsiness/model.py`](src/drowsiness/model.py). It uses standard convolutions, batch
normalization, SiLU activations, residual connections, max pooling, global average pooling,
dropout, and a two-logit linear classification head. The model has approximately 1.3 million
trainable parameters.

Training uses the following PyTorch features:

- `DataLoader` for batching the train and validation datasets.
- Weighted cross-entropy loss to account for class frequency.
- AdamW with a `3e-4` learning rate and `1e-4` weight decay.
- Cosine annealing for learning-rate scheduling.
- FP16 automatic mixed precision on ROCm.
- Seeded, stratified train/validation splitting and checkpointing by lowest validation loss.

Although PyTorch uses the `torch.cuda` API namespace, the container build is backed by AMD HIP and
ROCm rather than NVIDIA CUDA.

## Data and preprocessing stack

The project uses the Hugging Face dataset
[`akahana/Driver-Drowsiness-Dataset`](https://huggingface.co/datasets/akahana/Driver-Drowsiness-Dataset)
at the revision pinned in [`configs/mvp.yaml`](configs/mvp.yaml). Hugging Face Datasets manages the
download, cache, official train/test splits, and the seeded validation split.

Pillow decodes each source image, while TorchVision's v2 transforms provide:

- RGB conversion and resize to `224 x 224`.
- Random horizontal flip, mild color jitter, translation, and rotation during training.
- Float conversion and channel normalization to the `[-1, 1]` input range.
- Deterministic preprocessing for validation and testing.

The audit pipeline uses Pillow, hashing from Python's standard library, and `tqdm` to detect corrupt
images and exact duplicates before training.

## Evaluation and export stack

scikit-learn calculates accuracy, balanced accuracy, precision, recall, F1/F2, ROC-AUC, average
precision, Cohen's kappa, Matthews correlation coefficient, and the confusion matrix. It is also
used to choose the drowsy-score threshold on validation data by maximizing F2.

PyTorch exports the trained checkpoint to an opset-18 ONNX graph. The `onnx` package validates the
graph, and Python ONNX Runtime compares its logits and predicted classes with the original PyTorch
model before browser publication.

## Browser application stack

The browser app is a framework-free TypeScript application bundled by Vite. Its selector loads the
custom CNN or the legacy MobileNetV3 ONNX model. ONNX Runtime Web prefers WebGPU, with WebAssembly
as the CPU fallback. MediaPipe supplies face detection only when MobileNet is selected; the custom
CNN continues to use its fixed center crop. Native browser APIs handle the rest of the pipeline:

- MediaDevices for webcam access.
- Canvas 2D for the centered crop, resize, and pixel extraction.
- Web Audio for the warning sound.
- `requestAnimationFrame` for webcam inference scheduling.

Images, webcam frames, and prediction scores remain in the browser; the app does not require an
inference server.

## Environment and development tools

- The ROCm container is based on
  `rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1`.
- Docker Compose passes `/dev/kfd` and `/dev/dri` into the trainer container for AMD GPU access.
- GNU Make provides commands for installation, linting, tests, audits, training, evaluation,
  export, reports, and web builds.
- Pytest and pytest-cov are optional development dependencies for automated tests and coverage.
- Ruff performs Python linting and formatting checks.
- Node.js `>=22` and npm install the locked browser dependencies and run the Vite toolchain.
- `python-docx` and ReportLab are optional dependencies for generating DOCX and PDF technical
  reports; they are not required to train or run the model.

## End-to-end flow

```text
Hugging Face dataset
  -> Pillow decoding and dataset audit
  -> TorchVision preprocessing and augmentation
  -> custom PyTorch CNN training on AMD ROCm
  -> scikit-learn validation and test metrics
  -> ONNX export and ONNX Runtime parity check
  -> browser model selector
       -> custom CNN with fixed center crop
       -> legacy MobileNetV3 with MediaPipe face crop
  -> ONNX Runtime Web inference through WebGPU or WebAssembly
```

See [`README.md`](README.md) for the model architecture and full workflow, and
[`RUN_MODEL.md`](RUN_MODEL.md) for operating instructions.
