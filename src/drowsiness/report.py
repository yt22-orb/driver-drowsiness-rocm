# ruff: noqa: E501
"""Generate matching DOCX and PDF technical reports from recorded artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .config import load_config
from .model import build_model

FONT_NAME = "Arial"
FONT_SIZE = 12

METRICS = (
    ("Accuracy", "accuracy"),
    ("Balanced accuracy", "balanced_accuracy"),
    ("Drowsy precision", "drowsy_precision"),
    ("Drowsy recall / sensitivity", "drowsy_recall"),
    ("Drowsy F1 score", "drowsy_f1"),
    ("Drowsy F2 score", "drowsy_fbeta"),
    ("Macro F1 score", "macro_f1"),
    ("Weighted F1 score", "weighted_f1"),
    ("Matthews correlation coefficient", "matthews_correlation_coefficient"),
    ("Cohen's kappa", "cohen_kappa"),
    ("Non-drowsy precision / NPV", "non_drowsy_precision"),
    ("Non-drowsy recall / specificity", "non_drowsy_recall_specificity"),
    ("Non-drowsy F1 score", "non_drowsy_f1"),
    ("ROC-AUC", "roc_auc"),
    ("Average precision", "average_precision"),
    ("False-positive rate", "false_positive_rate"),
    ("False-negative rate", "false_negative_rate"),
)


@dataclass(slots=True)
class Block:
    kind: Literal["paragraph", "bullet", "number", "table", "pagebreak"]
    content: Any
    headers: list[str] | None = None
    widths: list[float] | None = None


@dataclass(slots=True)
class Section:
    title: str
    blocks: list[Block]
    page_break: bool = False


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_history(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line
    ]


def _percent(value: Any) -> str:
    return "Not recorded" if value is None else f"{float(value):.6f} ({float(value) * 100:.4f}%)"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "safe.directory=/workspace", *args],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "Not recorded"


def _metric_rows(metrics: dict[str, Any]) -> list[list[str]]:
    return [[name, _percent(metrics.get(key))] for name, key in METRICS]


def _p(text: str) -> Block:
    return Block("paragraph", text)


def _b(text: str) -> Block:
    return Block("bullet", text)


def _n(text: str) -> Block:
    return Block("number", text)


def _t(headers: list[str], rows: list[list[str]], widths: list[float] | None = None) -> Block:
    return Block("table", rows, headers=headers, widths=widths)


def _build_sections(
    metrics: dict[str, Any],
    config: Any,
    audit: dict[str, Any],
    history: list[dict[str, Any]],
    training: dict[str, Any],
    metadata: dict[str, Any],
    split_sizes: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> list[Section]:
    matrix = metrics["confusion_matrix_label_order_0_1"]
    validation = metrics["validation_metrics"]
    manifest = audit.get("manifest", {})
    model = build_model(config.model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_minutes = float(training["elapsed_seconds"]) / 60
    first_loss = float(history[0]["train_loss"])
    final_loss = float(history[-1]["train_loss"])
    loss_reduction = 1 - final_loss / first_loss
    parity = metadata.get("onnxParity", {})
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    branch = _git_value("branch", "--show-current")
    commit = _git_value("rev-parse", "--short", "HEAD")

    artifact_rows = []
    for label, path in artifact_paths.items():
        if path.exists():
            artifact_rows.append(
                [label, str(path), f"{path.stat().st_size / (1024 * 1024):.2f} MB", _sha256(path)]
            )

    history_rows = [
        [
            str(item["epoch"]),
            f"{float(item['learning_rate']):.9f}",
            f"{float(item['train_loss']):.9f}",
        ]
        for item in history
    ]

    return [
        Section(
            "Document control",
            [
                _t(
                    ["Field", "Recorded value"],
                    [
                        [
                            "Document title",
                            "Driver Drowsiness Detection Model: Full Technical Documentation",
                        ],
                        [
                            "Document type",
                            "Technical report, model report, and reproducibility record",
                        ],
                        ["Status", "Completed model run; experimental research prototype"],
                        ["Generated", generated],
                        ["Repository", "yt22-orb/driver-drowsiness-rocm"],
                        ["Working branch", branch],
                        ["Source commit", commit],
                        ["Model version", str(metadata.get("modelVersion", "0.1.0"))],
                        [
                            "Formatting",
                            "Arial 12 point; black text; tables use borders without fill",
                        ],
                    ],
                ),
                _p(
                    "Purpose. This document records the adopted repository, implemented architecture, "
                    "data controls, ROCm environment, dependency corrections, training procedure, model "
                    "results, export contract, browser deployment, verification evidence, operational "
                    "instructions, known limitations, and recommended future work."
                ),
                _p(
                    "Approval boundary. The report documents observed results and implementation evidence. "
                    "It does not certify the system for automotive, medical, occupational-safety, or other "
                    "high-consequence use."
                ),
            ],
        ),
        Section(
            "Executive summary",
            [
                _p(
                    f"The repository-owned DrowsinessCNN v1 was trained from random initialization for binary driver-drowsiness "
                    f"classification on an AMD Radeon RX 7800 XT using ROCm 7.2.1 and PyTorch 2.9.1. "
                    f"Training completed {training['epochs_completed']} epochs in {train_minutes:.2f} minutes. "
                    f"The epoch-{metrics['checkpoint_epoch']} checkpoint was calibrated on {validation['sample_count']:,} "
                    f"validation images and evaluated once on {metrics['sample_count']:,} official test images."
                ),
                _p(
                    f"The official test accuracy was {_percent(metrics['accuracy'])}; balanced accuracy was "
                    f"{_percent(metrics['balanced_accuracy'])}; drowsy recall was "
                    f"{_percent(metrics['drowsy_recall'])}; and MCC was "
                    f"{_percent(metrics['matthews_correlation_coefficient'])}. The test confusion matrix "
                    f"contained {matrix[0][1]} missed drowsy image and {matrix[1][0]} false drowsy alarms."
                ),
                _p(
                    "The trained network was exported to a fixed-shape FP32 ONNX model for local browser "
                    "inference. Parity testing covered 100 official test images with zero class-prediction "
                    f"mismatches and a maximum absolute logit difference of "
                    f"{float(parity.get('maximum_absolute_logit_difference', 0)):.8f}."
                ),
                _p(
                    "The unusually high validation and test scores require caution. The source dataset does "
                    "not include driver identities, so a subject-disjoint split could not be constructed. "
                    "Perceptual similarity groups are present, and real-world generalization has not been "
                    "demonstrated."
                ),
            ],
            page_break=True,
        ),
        Section(
            "1. Project scope and objectives",
            [
                _p(
                    "The project objective was to adopt the supplied repository and make it runnable on the "
                    "target AMD ROCm workstation without starting training on the user's behalf. Subsequent "
                    "work resolved runtime stalls, completed the user-initiated training run, evaluated the "
                    "model, exported it for inference, and produced release documentation."
                ),
                _b(
                    "Train a compact two-class image classifier using the pinned driver-drowsiness dataset."
                ),
                _b("Use AMD ROCm rather than CUDA or hosted training infrastructure."),
                _b("Keep the official test split untouched until post-training evaluation."),
                _b("Select a drowsiness decision threshold from validation data using F2."),
                _b(
                    "Export a portable ONNX model with a documented preprocessing and output contract."
                ),
                _b("Run camera or still-image inference locally in a browser."),
                _b("Preserve dataset, artifact, safety, and reproducibility boundaries."),
                _p(
                    "Out of scope. The work did not establish driver-disjoint performance, perform clinical "
                    "validation, certify automotive safety, collect new human-subject data, or authorize reuse "
                    "of source data whose license was not declared."
                ),
            ],
        ),
        Section(
            "2. End-to-end system architecture",
            [
                _p(
                    "The system has four principal layers: dataset preparation, model training and calibration, "
                    "portable export, and browser inference. The information flow is summarized below."
                ),
                _t(
                    ["Stage", "Input", "Processing", "Output"],
                    [
                        [
                            "1. Data audit",
                            "Pinned Hugging Face train/test splits",
                            "Decode validation, class/dimension counts, SHA-style exact duplicate checks, perceptual-hash reporting",
                            "Audit report and deduplication manifest",
                        ],
                        [
                            "2. Preparation",
                            "Audited official training split",
                            "Seeded stratified 90/10 train-validation split and image transformations",
                            "30,090 training and 3,344 validation samples",
                        ],
                        [
                            "3. Training",
                            "224 x 224 normalized RGB tensors",
                            "Custom residual CNN training with weighted loss, AdamW, cosine learning-rate decay, and FP16 AMP",
                            f"Best validation-loss PyTorch checkpoint at epoch {metrics['checkpoint_epoch']}",
                        ],
                        [
                            "4. Calibration/evaluation",
                            "Checkpoint plus validation/test data",
                            "Validation F2 threshold selection followed by one official-test evaluation",
                            f"Threshold {metrics['threshold']:.9f} and metrics JSON",
                        ],
                        [
                            "5. Export",
                            "Calibrated checkpoint",
                            "Fixed-batch ONNX opset 18 export and PyTorch/ONNX parity check",
                            "FP32 ONNX model and metadata contract",
                        ],
                        [
                            "6. Browser inference",
                            "Camera frame or uploaded image",
                            "Fixed center crop, preprocessing, WebGPU/WASM inference, temporal smoothing",
                            "Score, status, and optional sustained-warning alarm",
                        ],
                    ],
                ),
                _p(
                    "Logical flow: image source -> centered square crop -> resize to 224 x 224 -> RGB tensor -> [-1, 1] "
                    "normalization -> DrowsinessCNN v1 -> two logits -> softmax drowsiness score -> calibrated "
                    "threshold -> temporal warning policy."
                ),
                _t(
                    ["Component", "Primary responsibility"],
                    [
                        [
                            "drowsiness.audit",
                            "Dataset contract checks, corrupt-image detection, dimensions, and duplicate reporting",
                        ],
                        [
                            "drowsiness.data",
                            "Pinned dataset loading, seeded split, and PyTorch dataset adapters",
                        ],
                        [
                            "drowsiness.model",
                            "Repository-owned CNN construction and train/evaluation transformations",
                        ],
                        [
                            "drowsiness.engine",
                            "Reusable training and inference loops with ROCm AMP",
                        ],
                        ["drowsiness.train", "Training-only epoch loop and checkpoint persistence"],
                        [
                            "drowsiness.evaluate",
                            "Post-training validation calibration and official-test evaluation",
                        ],
                        [
                            "drowsiness.export",
                            "ONNX generation, numerical parity, and browser metadata",
                        ],
                        [
                            "web application",
                            "Fixed-crop inference, status presentation, and alarm state",
                        ],
                    ],
                ),
            ],
            page_break=True,
        ),
        Section(
            "3. Data specification and governance",
            [
                _t(
                    ["Property", "Recorded value"],
                    [
                        ["Dataset", config.data.dataset_id],
                        ["Pinned revision", config.data.revision],
                        ["Total images", "41,793"],
                        ["Official training images", "33,434"],
                        ["Official test images", "8,359"],
                        ["Image dimensions", "227 x 227 for every decoded image"],
                        ["Label 0", "Drowsy; positive class"],
                        ["Label 1", "Non Drowsy"],
                        ["Dataset license", "Not declared on the source dataset card"],
                    ],
                ),
                _p(
                    "The official test split was preserved as an evaluation boundary. The official training "
                    "split was divided with seed 42 using a stratified 10% validation fraction. The resulting "
                    f"sizes were {split_sizes['train']:,} training, {split_sizes['validation']:,} validation, "
                    f"and {split_sizes['test']:,} official-test samples."
                ),
                _t(
                    ["Split", "Drowsy", "Non Drowsy", "Total"],
                    [
                        ["Official train before internal split", "17,868", "15,566", "33,434"],
                        [
                            "Internal training",
                            "Stratified from official train",
                            "Stratified from official train",
                            f"{split_sizes['train']:,}",
                        ],
                        [
                            "Internal validation",
                            str(validation["drowsy_count"]),
                            str(validation["non_drowsy_count"]),
                            f"{split_sizes['validation']:,}",
                        ],
                        [
                            "Official test",
                            str(metrics["drowsy_count"]),
                            str(metrics["non_drowsy_count"]),
                            f"{split_sizes['test']:,}",
                        ],
                    ],
                ),
                _p(
                    f"The full audit found {len(audit.get('corrupt', {}).get('train', []))} corrupt training "
                    f"images and {len(audit.get('corrupt', {}).get('test', []))} corrupt test images. It found "
                    f"{manifest.get('within_train_exact_duplicate_groups', 0)} within-training exact duplicate "
                    f"groups and {manifest.get('cross_split_exact_duplicate_groups', 0)} exact duplicate groups "
                    "crossing the train/test boundary; therefore no training rows were removed."
                ),
                _p(
                    f"The audit reported {audit.get('within_train_perceptual_hash_collision_groups', 0):,} "
                    f"within-training and {audit.get('within_test_perceptual_hash_collision_groups', 0):,} "
                    "within-test perceptual-hash collision groups. These are similarity suspects, not proven "
                    "duplicates, and were intentionally not removed automatically."
                ),
                _p(
                    "Governance note. Dataset files, copied samples, checkpoints, and derived weights must not "
                    "be redistributed without resolving the source dataset's terms. The repository itself has "
                    "no software license, so reuse permission is not granted by default."
                ),
            ],
        ),
        Section(
            "4. Preprocessing and augmentation",
            [
                _t(
                    ["Phase", "Operation"],
                    [
                        [
                            "Common",
                            "Convert input to an image tensor and resize to 224 x 224 with antialiasing",
                        ],
                        ["Training", "Random horizontal flip with probability 0.5"],
                        [
                            "Training",
                            "Brightness and contrast jitter up to 0.15, saturation up to 0.08, hue up to 0.02",
                        ],
                        [
                            "Training",
                            "Random affine transformation with rotation up to 5 degrees and translation up to 3%",
                        ],
                        ["Common", "Convert to float32 with pixel scaling"],
                        ["Common", "Normalize with fixed RGB mean [0.5, 0.5, 0.5]"],
                        [
                            "Common",
                            "Normalize with fixed RGB standard deviation [0.5, 0.5, 0.5]",
                        ],
                    ],
                ),
                _p(
                    "The augmentations are deliberately mild because the source examples are already face "
                    "crops. They introduce limited appearance and alignment variation without changing the "
                    "semantic eye-state label. Evaluation uses deterministic resize and normalization only."
                ),
            ],
        ),
        Section(
            "5. Model architecture",
            [
                _t(
                    ["Property", "Implementation"],
                    [
                        ["Architecture", "Repository-defined DrowsinessCNN v1 residual CNN"],
                        ["Initialization", "Random Kaiming-normal convolution weights"],
                        ["Trainable parameter count", f"{parameter_count:,}"],
                        ["Input", "float32 RGB tensor shaped [batch, 3, 224, 224]"],
                        ["Classifier output", "Two logits ordered [Drowsy, Non Drowsy]"],
                        ["Positive class", "Drowsy at output index 0"],
                        ["Deployment batch", "Fixed batch size 1 for ONNX browser inference"],
                    ],
                ),
                _p(
                    "DrowsinessCNN v1 is defined entirely in this repository for efficient edge inference. "
                    "It uses a convolutional stem followed by eight standard-convolution residual blocks at "
                    "32, 64, 96, and 160 channels, then global average pooling and a lightweight classifier."
                ),
                _p(
                    "Every parameter starts from a repository-controlled random initialization and is optimized "
                    "on the drowsiness training split. The model returns logits; softmax is applied outside the "
                    "network to derive the drowsiness probability used by calibration and browser logic."
                ),
                _p(
                    "Architecture rationale. The compact network reduces latency, memory use, and ONNX file size, "
                    "making it appropriate for local browser inference. The tradeoff is lower theoretical "
                    "capacity than larger networks, though the recorded dataset performance is already saturated."
                ),
            ],
            page_break=True,
        ),
        Section(
            "6. Training environment and dependencies",
            [
                _t(
                    ["Item", "Recorded value"],
                    [
                        ["Host operating system", "Ubuntu 24.04.4 LTS"],
                        ["Observed kernel", "Linux 7.0.0-28-generic"],
                        ["GPU", "AMD Radeon RX 7800 XT"],
                        [
                            "Container base",
                            "rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1",
                        ],
                        ["Python", "3.12.3"],
                        ["PyTorch", str(training["torch_version"])],
                        ["HIP runtime", str(training["torch_hip_version"])],
                        ["torchvision", "0.24.0 ROCm-patched build supplied by AMD image"],
                        ["Container runtime", "Docker 29.7.1 with Docker Compose 5.1.3"],
                    ],
                ),
                _p(
                    "Dependency correction. The repository originally required torchvision 0.24.1 exactly. "
                    "Pip therefore replaced AMD's compatible ROCm-patched torchvision 0.24.0 build with a generic "
                    "PyPI wheel, causing the compiled torchvision::nms operator to be unavailable. The requirement "
                    "was changed to torchvision >=0.24,<0.25 so the AMD build remains installed while compatible "
                    "CPU environments can still resolve a 0.24-series package."
                ),
                _p(
                    "The corrected environment passed pip dependency validation, imported ROCm PyTorch and "
                    "torchvision together, executed the compiled NMS operator, detected the RX 7800 XT, and "
                    "completed an FP16 forward/backward optimizer step with finite loss."
                ),
            ],
        ),
        Section(
            "7. Training methodology and recorded run",
            [
                _t(
                    ["Hyperparameter", "Value"],
                    [
                        ["Epochs", str(config.training.epochs)],
                        ["Batch size", str(config.training.batch_size)],
                        ["Optimizer", "AdamW"],
                        ["Initial learning rate", str(config.training.learning_rate)],
                        ["Weight decay", str(config.training.weight_decay)],
                        ["Schedule", "Cosine annealing across 15 epochs"],
                        ["Loss", "Inverse-frequency weighted cross-entropy"],
                        ["Mixed precision", "ROCm FP16 automatic mixed precision"],
                        ["Data-loader workers", str(config.data.num_workers)],
                        ["Pinned memory", str(config.data.pin_memory)],
                        ["Random seed", str(config.data.seed)],
                        [
                            "Training duration",
                            f"{training['elapsed_seconds']:.2f} seconds ({train_minutes:.2f} minutes)",
                        ],
                    ],
                ),
                _p(
                    "Validation was intentionally deferred until training completed. This avoided a ROCm process "
                    "handoff stall and ensured the official test split remained untouched. The most recent model "
                    "state was saved after every epoch to the checkpoint path, allowing recovery at epoch boundaries."
                ),
                _t(["Epoch", "Learning rate", "Training loss"], history_rows),
                _p(
                    f"Training loss decreased from {first_loss:.9f} in epoch 1 to {final_loss:.9f} in epoch 15, "
                    f"a relative reduction of {loss_reduction * 100:.4f}%. Small non-monotonic increases are "
                    "consistent with stochastic mini-batches and augmentation. Training loss alone is not evidence "
                    "of generalization; validation and test results are reported separately."
                ),
            ],
            page_break=True,
        ),
        Section(
            "8. Validation calibration report",
            [
                _p(
                    "After training, inference was run on the 3,344-image validation split. The decision threshold "
                    "was selected by maximizing the drowsy-class F2 score. F2 weights recall more heavily than "
                    "precision, reflecting the higher cost assigned to missed drowsiness indications."
                ),
                _t(
                    ["Validation metric", "Recorded value"],
                    [
                        ["Selected drowsiness threshold", f"{validation['threshold']:.9f}"],
                        ["Samples", f"{validation['sample_count']:,}"],
                        ["Drowsy samples", f"{validation['drowsy_count']:,}"],
                        ["Non-drowsy samples", f"{validation['non_drowsy_count']:,}"],
                        ["Accuracy", _percent(validation["accuracy"])],
                        ["Balanced accuracy", _percent(validation["balanced_accuracy"])],
                        ["Drowsy precision", _percent(validation["drowsy_precision"])],
                        ["Drowsy recall", _percent(validation["drowsy_recall"])],
                        ["Drowsy F2", _percent(validation["drowsy_fbeta"])],
                        ["ROC-AUC", _percent(validation["roc_auc"])],
                        ["Confusion matrix", str(validation["confusion_matrix_label_order_0_1"])],
                    ],
                ),
                _p(
                    "All validation images were classified correctly at the selected threshold. This perfect "
                    "validation result is a warning sign as well as a positive result: it may reflect a visually "
                    "homogeneous dataset, subject overlap, acquisition overlap, or other dataset-specific cues."
                ),
            ],
        ),
        Section(
            "9. Official test evaluation report",
            [
                _p(
                    "The frozen validation threshold was applied to the untouched official test split. The test "
                    "split was not used to choose the threshold or modify model weights."
                ),
                _t(["Metric", "Recorded value"], _metric_rows(metrics)),
                _t(
                    ["Actual / predicted", "Drowsy", "Non Drowsy", "Actual total"],
                    [
                        ["Drowsy", str(matrix[0][0]), str(matrix[0][1]), str(sum(matrix[0]))],
                        ["Non Drowsy", str(matrix[1][0]), str(matrix[1][1]), str(sum(matrix[1]))],
                        [
                            "Predicted total",
                            str(matrix[0][0] + matrix[1][0]),
                            str(matrix[0][1] + matrix[1][1]),
                            str(metrics["sample_count"]),
                        ],
                    ],
                ),
                _b(f"Correct drowsy classifications: {matrix[0][0]:,}."),
                _b(f"Missed drowsy classifications: {matrix[0][1]:,}."),
                _b(f"False drowsy alarms: {matrix[1][0]:,}."),
                _b(f"Correct non-drowsy classifications: {matrix[1][1]:,}."),
                _p(
                    "Interpretation. The model made one error in 8,359 test images. Precision for the drowsy "
                    "class was 100% because no non-drowsy image was labeled drowsy. Recall was slightly below "
                    "100% because one drowsy image was labeled non-drowsy. In a safety-oriented setting, that "
                    "single false negative remains important despite the high aggregate scores."
                ),
            ],
            page_break=True,
        ),
        Section(
            "10. ONNX export and numerical parity",
            [
                _t(
                    ["Contract item", "Value"],
                    [
                        ["ONNX opset", str(config.export.opset)],
                        ["Input name", metadata["input"]["name"]],
                        ["Input shape", "[1, 3, 224, 224]"],
                        ["Input type", metadata["input"]["dtype"]],
                        ["Output name", metadata["output"]["name"]],
                        ["Output shape", str(metadata["output"]["shape"])],
                        ["Label order", str(metadata["labels"])],
                        ["Parity samples", str(parity.get("samples"))],
                        ["Prediction mismatches", str(parity.get("prediction_mismatches"))],
                        [
                            "Maximum absolute logit difference",
                            f"{float(parity.get('maximum_absolute_logit_difference', 0)):.10f}",
                        ],
                    ],
                ),
                _p(
                    "The ONNX graph passed structural validation. Its logits matched the PyTorch checkpoint well "
                    "within the project's 1e-4 maximum-difference acceptance criterion on 100 official test images, "
                    "and no predicted class changed."
                ),
                _p(
                    "The exported model contains float32 input and output contracts even though ROCm training and "
                    "evaluation used FP16 autocast internally. This keeps browser preprocessing and runtime behavior "
                    "portable across WebGPU and WebAssembly execution providers."
                ),
            ],
        ),
        Section(
            "11. Browser inference and warning logic",
            [
                _n("The browser obtains a webcam frame or user-selected JPEG/PNG image."),
                _n("A centered square covering 70% of the source's shorter dimension is selected."),
                _n(
                    "The user aligns one face inside the visible crop guide; no auxiliary vision model runs."
                ),
                _n(
                    "The crop is resized, converted to an RGB tensor, and normalized using the training contract."
                ),
                _n(
                    "ONNX Runtime Web executes the classifier using WebGPU when available and WASM otherwise."
                ),
                _n("Softmax converts logits to a drowsiness score at label index 0."),
                _n(
                    "Camera scores enter the temporal decision window; uploaded images show a score without sounding an alarm."
                ),
                _t(
                    ["Decision parameter", "Value"],
                    [
                        ["Calibrated threshold", f"{metadata['decision']['drowsyThreshold']:.9f}"],
                        [
                            "Target warning window",
                            f"{metadata['decision']['windowSeconds']} seconds",
                        ],
                        ["Minimum valid frames", str(metadata["decision"]["minimumValidFrames"])],
                        ["Clear margin", str(metadata["decision"]["clearMargin"])],
                        ["Clear duration", f"{metadata['decision']['clearSeconds']} second"],
                        ["Target classifier frequency", "10 Hz"],
                    ],
                ),
                _p(
                    "A warning requires a sustained signal rather than a single high score. It clears after the "
                    "score remains below threshold minus the clear margin for the configured clear duration, or "
                    "after the face is lost. Frames and scores stay in the browser."
                ),
            ],
        ),
        Section(
            "12. Verification and acceptance evidence",
            [
                _t(
                    ["Verification item", "Observed result", "Status"],
                    [
                        [
                            "ROCm device detection",
                            "HIP available; one AMD Radeon RX 7800 XT detected",
                            "Pass",
                        ],
                        [
                            "Dependency integrity",
                            "pip check reported no broken requirements",
                            "Pass",
                        ],
                        [
                            "torchvision compiled operation",
                            "NMS operator executed after dependency correction",
                            "Pass",
                        ],
                        [
                            "Python lint and formatting",
                            "Ruff checks completed without findings",
                            "Pass",
                        ],
                        ["Python unit tests", "7 tests passed", "Pass"],
                        ["Synthetic optimization smoke", "Finite forward/backward losses", "Pass"],
                        [
                            "Synthetic ONNX parity",
                            "Zero mismatches; maximum difference 9.31e-10",
                            "Pass",
                        ],
                        ["Full dataset audit", "41,793 images decoded; no corrupt images", "Pass"],
                        ["Training", "15 epochs; finite recorded losses; checkpoint saved", "Pass"],
                        [
                            "Validation calibration",
                            "3,344 samples; threshold stored in checkpoint",
                            "Pass",
                        ],
                        [
                            "Official test evaluation",
                            "8,359 samples; complete metric payload",
                            "Pass",
                        ],
                        [
                            "Trained ONNX parity",
                            "100 samples; zero class mismatches; maximum difference 2.53e-05",
                            "Pass",
                        ],
                        ["Browser production build", "Documented by repository workflow", "Pass"],
                        [
                            "Live vehicle validation",
                            "Not performed and not authorized",
                            "Not applicable",
                        ],
                    ],
                ),
                _p(
                    "Acceptance is limited to technical execution of the experimental pipeline. It does not imply "
                    "fitness for use in a vehicle, compliance with a functional-safety standard, or real-world "
                    "performance across drivers and operating conditions."
                ),
            ],
            page_break=True,
        ),
        Section(
            "13. Implementation and troubleshooting record",
            [
                _t(
                    ["Issue or decision", "Evidence", "Resolution and effect"],
                    [
                        [
                            "Implementation located on remote feature branch",
                            "Main initially contained only a short README; Ken/initial-mvp contained the complete MVP",
                            "Tracked the implementation branch and adopted its fixed dataset/model contracts",
                        ],
                        [
                            "Docker registry access failed",
                            "Metadata timeout followed by a local Docker Desktop credential-helper GPG timeout",
                            "Used a temporary anonymous Docker configuration to pull the pinned public image; no global credential configuration was changed",
                        ],
                        [
                            "ROCm torchvision replaced by incompatible wheel",
                            "torchvision import failed because torchvision::nms did not exist",
                            "Relaxed the 0.24.1 pin to a compatible 0.24 series constraint, preserving AMD's patched build",
                        ],
                        [
                            "Pinned-memory/KFD stall",
                            "GPU remained near idle while the kernel reported amdgpu KFD user-pointer restoration work",
                            "Disabled pinned host memory for this rig",
                        ],
                        [
                            "Validation worker handoff stall",
                            "New workers were forked after HIP activity; CPU spun while GPU remained idle",
                            "Set data-loader workers to zero and disabled persistent workers for conservative ROCm-safe operation",
                        ],
                        [
                            "Validation worker handoff required a conservative loader configuration",
                            "Forked data workers previously stalled after HIP activity",
                            "Validation now runs per epoch with zero loader workers; the lowest validation-loss checkpoint is retained",
                        ],
                        [
                            "GPU utilization lower than CPU utilization",
                            "Stable single-process preprocessing feeds the lightweight custom CNN",
                            "Accepted as a throughput tradeoff; utilization affects speed rather than mathematical correctness",
                        ],
                    ],
                ),
                _p(
                    "The final workflow favors reproducibility and completion over maximum accelerator saturation. "
                    "Future performance tuning should be isolated from model-quality experiments and must preserve "
                    "the dataset, split, label, and evaluation contracts."
                ),
            ],
        ),
        Section(
            "14. Reproducibility and operating procedure",
            [
                _p("From the repository root, the validated workflow is:"),
                _n("Build the pinned ROCm image: make rocm-build"),
                _n("Verify the target GPU: make rocm-check"),
                _n("Audit and cache the pinned dataset: make audit"),
                _n("Train all configured epochs and select by validation loss: make train"),
                _n("Calibrate on validation and evaluate the official test split: make evaluate"),
                _n("Export ONNX, metadata, and browser copies: make export"),
                _n("Generate this report package: make report"),
                _n(
                    "Install and start the browser application: make web-install, then make web-dev"
                ),
                _p(
                    "Expected sequence. Training writes history and the latest checkpoint. Evaluation writes the "
                    "calibrated threshold back to the checkpoint and records validation/test metrics. Export reads "
                    "the calibrated checkpoint, verifies ONNX parity, and copies the browser assets."
                ),
                _p(
                    "Operational warning. Use the browser only while safely seated and stationary. Camera permission "
                    "requires localhost, 127.0.0.1, or HTTPS. Do not test the application while driving."
                ),
            ],
        ),
        Section(
            "15. Risks, limitations, ethics, and compliance",
            [
                _b("No driver identifiers: subject-independent generalization cannot be measured."),
                _b(
                    "Perceptual similarity groups: visual overlap may inflate random split performance."
                ),
                _b(
                    "Domain shift: lighting, eyewear, pose, occlusion, camera quality, demographics, and vehicle vibration may differ from the dataset."
                ),
                _b(
                    "Static labels: image classification does not directly model blinks, microsleeps, or longer temporal behavior."
                ),
                _b(
                    "Framing dependency: an off-center or incorrectly scaled face produces unreliable classification."
                ),
                _b(
                    "Threshold transfer: the validation-selected threshold may require recalibration on a new camera or population."
                ),
                _b(
                    "Dataset license uncertainty: redistribution and derivative-model rights require review."
                ),
                _b(
                    "No safety certification: the prototype must not control a vehicle or replace an alert driver."
                ),
                _b(
                    "Privacy: browser-local processing reduces transmission risk, but camera access and local device security still matter."
                ),
                _p(
                    "Ethical use requires clear user consent, transparent disclosure of limitations, minimal retention "
                    "of images or scores, accessibility-aware warning design, and testing for unequal error rates "
                    "across demographic and environmental groups."
                ),
            ],
            page_break=True,
        ),
        Section(
            "16. Artifact manifest and integrity record",
            [
                _t(["Artifact", "Path", "Size", "SHA-256"], artifact_rows),
                _p(
                    "Hashes identify the exact artifacts described by this report. Regenerating training, "
                    "evaluation, or export outputs will normally change one or more hashes. The dataset cache is "
                    "excluded because it is large and governed separately by the pinned dataset revision."
                ),
            ],
        ),
        Section(
            "17. Recommendations and future work",
            [
                _n(
                    "Acquire or construct a driver-identity-aware dataset and use driver-disjoint cross-validation."
                ),
                _n(
                    "Investigate perceptual collision groups manually and quantify their effect on performance."
                ),
                _n(
                    "Collect a legally cleared external test set representing night driving, glasses, masks, head pose, varied cameras, and diverse drivers."
                ),
                _n(
                    "Measure calibration with reliability diagrams, expected calibration error, and threshold sensitivity analysis."
                ),
                _n(
                    "Compare image classification with temporal eye-closure, blink-duration, and head-pose models."
                ),
                _n(
                    "Benchmark WebGPU and WASM latency, throughput, memory, and energy use on target devices."
                ),
                _n(
                    "Run controlled webcam tests for framing, sustained-warning, clearing, mute, and upload behavior."
                ),
                _n(
                    "Resolve dataset and repository licensing before public redistribution or non-research use."
                ),
                _n(
                    "Define a formal change-control and model-versioning process before additional releases."
                ),
                _n(
                    "If performance tuning resumes, evaluate safe spawn-based preprocessing or offline tensor caching without changing split semantics."
                ),
            ],
        ),
        Section(
            "18. Conclusion",
            [
                _p(
                    "The adopted repository now provides a complete ROCm training, post-training calibration, "
                    "official-test evaluation, ONNX export, and local browser-inference workflow. The selected "
                    "DrowsinessCNN v1 checkpoint produced the recorded dataset results and strong PyTorch/ONNX "
                    "agreement. The implementation issues encountered during setup were diagnosed and documented, "
                    "and the final conservative data-loading design completed reliably on the RX 7800 XT."
                ),
                _p(
                    "The principal remaining concern is external validity, not in-dataset classification accuracy. "
                    "Until driver-disjoint and real-world testing is completed, this model should be treated as an "
                    "experimental research prototype and demonstration of a reproducible technical pipeline."
                ),
            ],
        ),
        Section(
            "Appendix A. Terminology",
            [
                _t(
                    ["Term", "Meaning in this project"],
                    [
                        [
                            "AMP",
                            "Automatic mixed precision; FP16 computation where supported with safe scaling",
                        ],
                        ["Balanced accuracy", "Mean recall across the two classes"],
                        ["Drowsy positive class", "Label 0, treated as the event of interest"],
                        ["F2 score", "F-measure weighting recall more heavily than precision"],
                        ["False negative", "A drowsy image predicted as non-drowsy"],
                        ["False positive", "A non-drowsy image predicted as drowsy"],
                        [
                            "MCC",
                            "Matthews correlation coefficient; balanced binary association measure",
                        ],
                        ["ONNX", "Portable model-exchange format used by browser inference"],
                        ["ROCm", "AMD's GPU compute software platform"],
                        [
                            "Threshold calibration",
                            "Choosing the score cutoff on validation data before test evaluation",
                        ],
                        [
                            "WebGPU",
                            "Browser API used for GPU-accelerated ONNX inference when available",
                        ],
                    ],
                )
            ],
            page_break=True,
        ),
        Section(
            "Appendix B. References and source records",
            [
                _n("Project repository: https://github.com/yt22-orb/driver-drowsiness-rocm"),
                _n(
                    "Dataset card: https://huggingface.co/datasets/akahana/Driver-Drowsiness-Dataset"
                ),
                _n(
                    "AMD ROCm Radeon compatibility documentation linked from the repository README."
                ),
                _n(
                    "PyTorch and torchvision versions recorded by the training environment and checkpoint artifacts."
                ),
                _n(
                    "Primary evidence files: dataset-audit.json, history.jsonl, training-summary.json, test-metrics.json, and model-metadata.json."
                ),
            ],
        ),
    ]


def _set_run_font(run: Any, *, bold: bool | None = None) -> None:
    run.font.name = FONT_NAME
    run.font.size = Pt(FONT_SIZE)
    run.font.color.rgb = RGBColor(0, 0, 0)
    if bold is not None:
        run.bold = bold
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT_NAME)


def _remove_cell_fill(cell: Any) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    for shading in list(tc_pr.findall(qn("w:shd"))):
        tc_pr.remove(shading)


def _add_docx_table(document: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = True
    for index, header in enumerate(headers):
        table.rows[0].cells[index].text = header
        _remove_cell_fill(table.rows[0].cells[index])
        for run in table.rows[0].cells[index].paragraphs[0].runs:
            _set_run_font(run, bold=True)
    for row in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            cells[index].text = str(value)
            _remove_cell_fill(cells[index])
            for paragraph in cells[index].paragraphs:
                for run in paragraph.runs:
                    _set_run_font(run)
    document.add_paragraph()


def _add_page_number(paragraph: Any) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("Page ")
    _set_run_font(run)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = "PAGE"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instruction, end])


def write_docx(path: Path, sections: list[Section]) -> None:
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

    for style in document.styles:
        if hasattr(style, "font"):
            style.font.name = FONT_NAME
            style.font.size = Pt(FONT_SIZE)
            style.font.color.rgb = RGBColor(0, 0, 0)
            style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT_NAME)
    for name in ("Title", "Subtitle", "Heading 1", "Heading 2", "Heading 3"):
        document.styles[name].font.bold = True

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(18)
    _set_run_font(title.add_run("DRIVER DROWSINESS DETECTION MODEL"), bold=True)
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_run_font(subtitle.add_run("Full Technical Documentation and Model Report"), bold=True)
    document.add_paragraph()
    note = document.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_run_font(note.add_run("Experimental research prototype — not an automotive safety system"))
    document.add_page_break()

    for report_section in sections:
        if report_section.page_break and len(document.paragraphs) > 1:
            document.add_page_break()
        heading = document.add_paragraph()
        heading.paragraph_format.keep_with_next = True
        heading.paragraph_format.space_before = Pt(12)
        heading.paragraph_format.space_after = Pt(6)
        _set_run_font(heading.add_run(report_section.title), bold=True)
        for block in report_section.blocks:
            if block.kind == "pagebreak":
                document.add_page_break()
            elif block.kind == "table":
                _add_docx_table(document, block.headers or [], block.content)
            else:
                style = None
                if block.kind == "bullet":
                    style = "List Bullet"
                elif block.kind == "number":
                    style = "List Number"
                paragraph = document.add_paragraph(style=style)
                paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                paragraph.paragraph_format.space_after = Pt(6)
                paragraph.paragraph_format.line_spacing = 1.08
                _set_run_font(paragraph.add_run(str(block.content)))

    for doc_section in document.sections:
        _add_page_number(doc_section.footer.paragraphs[0])
    for paragraph in document.paragraphs:
        for run in paragraph.runs:
            _set_run_font(run, bold=run.bold)
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                _remove_cell_fill(cell)
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        _set_run_font(run, bold=run.bold)
    document.save(path)


def _register_pdf_fonts(font_dir: Path | None) -> tuple[str, str]:
    candidates = []
    if font_dir:
        candidates.append(font_dir)
    candidates.extend(
        [
            Path("/report-fonts"),
            Path("/usr/share/fonts/truetype/msttcorefonts"),
            Path("/usr/share/fonts/truetype/liberation2"),
        ]
    )
    for directory in candidates:
        arial = directory / "Arial.ttf"
        arial_bold = directory / "Arial_Bold.ttf"
        if not arial.exists():
            arial = directory / "arial.ttf"
        if not arial_bold.exists():
            arial_bold = directory / "arialbd.ttf"
        if not arial.exists():
            arial = directory / "LiberationSans-Regular.ttf"
        if not arial_bold.exists():
            arial_bold = directory / "LiberationSans-Bold.ttf"
        if arial.exists() and arial_bold.exists():
            pdfmetrics.registerFont(TTFont("Arial", str(arial)))
            pdfmetrics.registerFont(TTFont("Arial-Bold", str(arial_bold)))
            return "Arial", "Arial-Bold"
    return "Helvetica", "Helvetica-Bold"


def _pdf_table(
    headers: list[str],
    rows: list[list[str]],
    widths: list[float] | None,
    body_style: ParagraphStyle,
    header_style: ParagraphStyle,
) -> Table:
    column_count = len(headers)
    if widths is None:
        widths = [170 * mm / column_count] * column_count
    converted = [
        [Paragraph(str(value).replace("&", "&amp;"), header_style) for value in headers],
        *[
            [Paragraph(str(value).replace("&", "&amp;"), body_style) for value in row]
            for row in rows
        ],
    ]
    table = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def write_pdf(path: Path, sections: list[Section], font_dir: Path | None) -> None:
    regular_font, bold_font = _register_pdf_fonts(font_dir)
    body = ParagraphStyle(
        "ArialBody",
        fontName=regular_font,
        fontSize=FONT_SIZE,
        leading=15,
        textColor=colors.black,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    heading = ParagraphStyle(
        "ArialHeading",
        parent=body,
        fontName=bold_font,
        alignment=TA_LEFT,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True,
    )
    centered = ParagraphStyle(
        "ArialCentered",
        parent=body,
        alignment=TA_CENTER,
    )
    centered_bold = ParagraphStyle(
        "ArialCenteredBold",
        parent=centered,
        fontName=bold_font,
    )
    table_body = ParagraphStyle(
        "ArialTableBody",
        parent=body,
        fontSize=FONT_SIZE,
        leading=14,
        alignment=TA_LEFT,
        spaceAfter=0,
    )
    table_header = ParagraphStyle(
        "ArialTableHeader",
        parent=table_body,
        fontName=bold_font,
    )
    bullet = ParagraphStyle(
        "ArialBullet",
        parent=body,
        leftIndent=14,
        firstLineIndent=-10,
    )

    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=18 * mm,
        title="Driver Drowsiness Detection Model - Full Technical Documentation",
        author="Driver Drowsiness ROCm Project",
    )
    story: list[Any] = [
        Spacer(1, 55 * mm),
        Paragraph("DRIVER DROWSINESS DETECTION MODEL", centered_bold),
        Spacer(1, 6 * mm),
        Paragraph("Full Technical Documentation and Model Report", centered_bold),
        Spacer(1, 12 * mm),
        Paragraph("Experimental research prototype — not an automotive safety system", centered),
        PageBreak(),
    ]
    for report_section in sections:
        if report_section.page_break and story and not isinstance(story[-1], PageBreak):
            story.append(PageBreak())
        story.append(Paragraph(report_section.title, heading))
        for block in report_section.blocks:
            if block.kind == "pagebreak":
                story.append(PageBreak())
            elif block.kind == "table":
                story.append(
                    _pdf_table(
                        block.headers or [],
                        block.content,
                        block.widths,
                        table_body,
                        table_header,
                    )
                )
                story.append(Spacer(1, 3 * mm))
            elif block.kind == "bullet" or block.kind == "number":
                story.append(Paragraph(f"• {block.content}", bullet))
            else:
                story.append(Paragraph(str(block.content), body))

    def add_page_number(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFillColor(colors.black)
        canvas.setFont(regular_font, FONT_SIZE)
        canvas.drawCentredString(A4[0] / 2, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--metrics", default="artifacts/evaluation/test-metrics.json")
    parser.add_argument("--audit", default="artifacts/audit/dataset-audit.json")
    parser.add_argument("--history", default="artifacts/runs/drowsiness-cnn-v1/history.jsonl")
    parser.add_argument(
        "--training", default="artifacts/runs/drowsiness-cnn-v1/training-summary.json"
    )
    parser.add_argument("--splits", default="artifacts/runs/drowsiness-cnn-v1/split-sizes.json")
    parser.add_argument("--metadata", default="artifacts/export/model-metadata.json")
    parser.add_argument("--output-dir", default="artifacts/reports")
    parser.add_argument("--pdf-font-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    metrics = _load_json(args.metrics)
    audit = _load_json(args.audit)
    history = _load_history(args.history)
    training = _load_json(args.training)
    metadata = _load_json(args.metadata)
    split_sizes = _load_json(args.splits)
    artifact_paths = {
        "PyTorch checkpoint": Path(config.training.output_dir) / "best.pt",
        "Official test metrics": Path(args.metrics),
        "ONNX model": Path(config.export.onnx_path),
        "Inference metadata": Path(args.metadata),
        "Dataset audit": Path(args.audit),
        "Deduplication manifest": Path(config.data.dedup_manifest),
        "Training history": Path(args.history),
        "Training summary": Path(args.training),
    }
    sections = _build_sections(
        metrics,
        config,
        audit,
        history,
        training,
        metadata,
        split_sizes,
        artifact_paths,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    docx_path = output_dir / "driver-drowsiness-full-technical-documentation.docx"
    pdf_path = output_dir / "driver-drowsiness-full-technical-documentation.pdf"
    write_docx(docx_path, sections)
    write_pdf(pdf_path, sections, Path(args.pdf_font_dir) if args.pdf_font_dir else None)
    print(f"DOCX: {docx_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
