# Driver Drowsiness ROCm

An experimental driver-drowsiness image classifier built around this repository's own compact
convolutional neural network. The network is initialized randomly, trained on the pinned
[Driver Drowsiness Dataset](https://huggingface.co/datasets/akahana/Driver-Drowsiness-Dataset),
exported to ONNX, and run locally in a browser.

The training environment targets an AMD Radeon RX 7800 XT with ROCm 7.2.1 and PyTorch 2.9.1.

> [!WARNING]
> This is an experimental thesis prototype, not a certified automotive safety system. Do not rely
> on it to decide whether it is safe to drive. Never test it while driving.

## Self-contained custom-model boundary

The current training and export pipeline contains one learned model: `DrowsinessCNN` from
[`src/drowsiness/model.py`](src/drowsiness/model.py). It does not load an external backbone,
detector, landmark estimator, feature extractor, or third-party checkpoint. Custom checkpoints and
browser metadata carry an architecture/weight-origin marker, and evaluation/export reject artifacts
that were not produced by this trainer.

The source dataset contains 227 × 227 face crops, so the model is a face-crop classifier rather
than a general scene detector. Browser inference uses a visible, deterministic center crop; the user
must keep one face centered and filling that guide. This limitation is intentional and avoids a
hidden second model in the custom pipeline. The browser's explicitly labeled legacy option is a
separate comparison path: it runs the earlier MobileNetV3 release with MediaPipe face detection.

## Custom architecture

`drowsiness_cnn_v1` is defined from ordinary PyTorch layers and has approximately 1.3 million
trainable parameters:

```text
RGB [N, 3, 224, 224]
  -> 5x5 Conv(3 -> 24), stride 2 + BatchNorm + SiLU
  -> 3x3 MaxPool, stride 2
  -> 2 residual blocks at 32 channels
  -> 2 residual blocks at 64 channels  (first block downsamples)
  -> 2 residual blocks at 96 channels  (first block downsamples)
  -> 2 residual blocks at 160 channels (first block downsamples)
  -> global average pooling
  -> dropout(0.25)
  -> Linear(160 -> 2 logits)
```

Each residual block contains two standard 3 × 3 convolutions. A learned 1 × 1 shortcut is used when
the spatial size or channel count changes. Convolutions use random Kaiming-normal initialization;
batch-normalization scales start at one, offsets at zero, and the final linear layer starts from a
small random normal distribution. No weights are downloaded during model construction.

## Dataset and labels

| Property | Value |
| --- | --- |
| Dataset | `akahana/Driver-Drowsiness-Dataset` |
| Revision | `1770cfcacac05ff4ae280479ed610c3fcc6b4b7c` |
| Official train rows | 33,434 |
| Official test rows | 8,359 |
| Label `0` | `Drowsy` (positive class) |
| Label `1` | `Non Drowsy` |

The repository pins the revision and preserves the official test split. The dataset card does not
declare a license; review its terms before redistributing data or derived weights.

## Training process

The complete workflow is:

```bash
make rocm-build
make rocm-check
make audit
make train
make evaluate
make export
make web-build
```

Training performs a seeded, stratified 90/10 split of the audited official training split. Exact
duplicates identified by the audit manifest are removed from training before the split. Training
augmentation consists of horizontal flip, mild color jitter, translation, and ±5° rotation.
Evaluation preprocessing is deterministic.

The network is optimized from random initialization with inverse-frequency weighted cross entropy,
AdamW (`3e-4` learning rate, `1e-4` weight decay), cosine learning-rate decay, and FP16 automatic
mixed precision on ROCm. Validation loss is measured after every epoch. `best.pt` is the epoch with
the lowest validation loss; `last.pt` records the final epoch. The official test set is not used for
checkpoint selection.

After training, `make evaluate` chooses a drowsy-score threshold on validation data by maximizing
F2, then evaluates the frozen checkpoint and threshold once on the official test split. `make
export` validates the checkpoint provenance, exports fixed-batch FP32 ONNX, checks PyTorch/ONNX
parity, and copies the model plus metadata into `web/public/models/`.

Default outputs:

| Artifact | Path |
| --- | --- |
| Audit report | `artifacts/audit/dataset-audit.json` |
| Dedup manifest | `artifacts/audit/dedup-manifest.json` |
| Resolved config | `artifacts/runs/drowsiness-cnn-v1/resolved-config.json` |
| Epoch history | `artifacts/runs/drowsiness-cnn-v1/history.jsonl` |
| Best checkpoint | `artifacts/runs/drowsiness-cnn-v1/best.pt` |
| Last checkpoint | `artifacts/runs/drowsiness-cnn-v1/last.pt` |
| Test metrics | `artifacts/evaluation/test-metrics.json` |
| ONNX model | `artifacts/export/drowsiness-cnn-v1.onnx` |
| Inference metadata | `artifacts/export/model-metadata.json` |

Generated datasets, weights, reports, and exports are ignored by Git.

## Inference input contract

Training, evaluation, Python parity checks, and browser inference share this tensor contract:

| Field | Contract |
| --- | --- |
| Source | One centered, tightly framed driver face in RGB |
| Browser crop | Centered square, side = 70% of the shorter source dimension |
| Resize | 224 × 224 |
| Tensor | float32 NCHW `[1, 3, 224, 224]` |
| Channel order | RGB |
| Scaling | `x = ((uint8 / 255) - 0.5) / 0.5`, producing `[-1, 1]` |
| Output | float32 logits `[1, 2]` in `[Drowsy, Non Drowsy]` order |
| Score | softmax probability at label index 0 |

For the webcam, place one face inside and fill the dashed square. For a still image, supply a JPEG
or PNG in which one face is centered in the same region. Frames without a correctly positioned face
are out of distribution; the app cannot identify or reject them because it deliberately uses no
auxiliary detector.

## Run browser inference

Train, evaluate, and export the custom model first. Then:

```bash
make web-install
make web-dev
```

Open <http://127.0.0.1:5173>. The app uses WebGPU when available and ONNX Runtime WebAssembly as a
fallback. It classifies the displayed crop at a target 10 Hz. A warning requires a sustained score
over the exported threshold; still-image tests show one score and never sound the alarm. Images,
frames, and scores stay in the browser.

The model selector can also run the earlier MobileNetV3-Small release alongside the custom CNN.
That legacy option retains its original MediaPipe face-detection crop and ImageNet normalization;
the custom CNN retains its fixed center-crop contract. See [`RUN_MODEL.md`](RUN_MODEL.md) for the
required local asset filenames.

See [`RUN_MODEL.md`](RUN_MODEL.md) for the short operating guide.

## CPU development

Python 3.12 or 3.13 is required:

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

`make smoke` performs a forward/backward optimizer step on synthetic images, exports a temporary
random-weight ONNX graph, validates it with `onnx.checker`, and checks numerical parity. The smoke
artifact is not a trained classifier.

## ROCm notes

The container expects `/dev/kfd` and `/dev/dri`. PyTorch retains the `torch.cuda` API name when it is
built for ROCm; `make rocm-check` verifies the HIP build and requires a device name containing
`7800 XT`. The image deliberately preserves AMD's ROCm-patched PyTorch/TorchVision packages.

If training runs out of memory, reduce the batch size from 128 to 64, then 32, and record the change.
Do not change the image size, architecture, labels, or official test boundary as an OOM workaround.
If ONNX parity fails, do not publish the export; check preprocessing, eval mode, opset, output order,
and runtime versions first.

## Verification checklist

- [ ] Audit completes with no unexplained corrupt images.
- [ ] Training starts from the custom architecture's random initialization.
- [ ] History contains finite training and validation loss for every epoch.
- [ ] `best.pt` has `checkpoint_format=drowsiness-cnn-v1` and
  `weights_origin=trained_from_scratch`.
- [ ] Threshold calibration uses validation data only.
- [ ] Official-test metrics are recorded without claiming real-world certification.
- [ ] ONNX parity passes on 100 official test images.
- [ ] Browser WebGPU and WASM paths both load schema-v2 metadata and the custom ONNX file.
- [ ] Webcam and image input are tested with the documented center-crop framing.

The source dataset has no subject identifiers, so driver-disjoint generalization cannot be measured
from the supplied data. Lighting, eyewear, pose, occlusion, camera quality, demographics, and vehicle
motion can all cause domain shift. Dataset accuracy alone is not evidence of safe on-road behavior.
