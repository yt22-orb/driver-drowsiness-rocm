## Driver drowsiness model v0.1.0

This release contains the trained epoch-15 MobileNetV3-Small checkpoint and the browser-ready FP32
ONNX export. Follow [`RUN_MODEL.md`](https://github.com/yt22-orb/driver-drowsiness-rocm/blob/agent/runnable-model-release/RUN_MODEL.md)
to try it with a local camera or uploaded image.

### Test results

- Official test samples: 8,359
- Accuracy: 99.9880%
- Balanced accuracy: 99.9888%
- Drowsy precision: 100.0000%
- Drowsy recall: 99.9777%
- Drowsy F1: 99.9888%
- Matthews correlation coefficient: 99.9760%
- ROC-AUC: 100.0000%
- Confusion matrix `[Drowsy, Non Drowsy]`: `[[4479, 1], [0, 3879]]`
- ONNX parity: 100 samples, zero prediction mismatches, maximum absolute logit difference 2.53e-05

### Assets

- `best.pt`: original PyTorch checkpoint with calibrated decision threshold
- `drowsiness-mobilenet-v3-small.onnx`: browser-ready model weights
- `model-metadata.json`: preprocessing, labels, threshold, metrics, and ONNX contract
- `test-metrics.json`: exact validation and official-test metric payload
- `driver-drowsiness-model-report.docx`: editable findings report
- `driver-drowsiness-model-report.pdf`: PDF findings report

### Important limitations

This is an experimental thesis prototype, not a certified automotive safety system. Never test it
while driving. The dataset does not provide subject identifiers, so driver-disjoint generalization
has not been established. The source dataset card declares no license; these derived weights are
provided for research and evaluation without granting rights to the source dataset. Review and
resolve the source dataset's terms before redistribution or non-research use.
