## Driver drowsiness custom CNN v1.0.0

This release contains the repository-defined `drowsiness_cnn_v1`, trained from random
initialization. It is separate from the earlier pretrained MobileNetV3 release. The accompanying
browser app can select either model and applies each model's original preprocessing contract.

### Recorded run

- Best checkpoint: epoch 15; validation loss `0.000013302130465320622`
- Official test set: 8,359 images
- Accuracy, balanced accuracy, drowsy precision/recall/F1, MCC, and ROC-AUC: `1.0`
- Confusion matrix `[Drowsy, Non Drowsy]`: `[[4480, 0], [0, 3879]]`
- Calibrated drowsy threshold: `0.9966691136360168`
- ONNX parity: 100 samples, 0 prediction mismatches, maximum logit difference `6.198883056640625e-06`
- Hardware: AMD Radeon RX 7800 XT
- Runtime: ROCm 7.2.1, PyTorch `2.9.1+rocm7.2.1.gitff65f5bc`

### Assets

- `drowsiness-cnn-v1-best.pt`: repository checkpoint marked `trained_from_scratch`
- `drowsiness-cnn-v1.onnx`: browser-ready custom CNN weights
- `model-metadata.json`: schema-v2 architecture, preprocessing, labels, threshold, and metrics
- `test-metrics.json`: exact validation and official-test metric payload
- `driver-drowsiness-full-technical-documentation.docx` and `.pdf`: generated technical report
- `drowsiness-cnn-v1-v1.0.0.tar.gz`: bundle containing the model artifacts above
- `SHA256SUMS`: hashes for every uploaded release asset except the checksum file itself

### Important limitations

This is an experimental thesis prototype, not a certified automotive safety system. The source
dataset contains face crops and does not provide subject identifiers, so driver-disjoint or full-scene
generalization has not been established. The browser requires the face to be centered inside its
fixed crop guide. Resolve the source dataset's terms before distributing derived weights.

The perfect result on this dataset is not evidence of real-world reliability and should be treated
as a prompt for leakage, domain-shift, and driver-disjoint follow-up evaluation.
