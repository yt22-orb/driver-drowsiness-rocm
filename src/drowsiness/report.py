"""Generate matching DOCX and PDF evaluation reports from recorded artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .config import load_config

METRICS = (
    ("Accuracy", "accuracy"),
    ("Balanced accuracy", "balanced_accuracy"),
    ("Drowsy precision", "drowsy_precision"),
    ("Drowsy recall / sensitivity", "drowsy_recall"),
    ("Drowsy F1 score", "drowsy_f1"),
    ("Drowsy F2 score", "drowsy_fbeta"),
    ("Macro F1 score", "macro_f1"),
    ("Weighted F1 score", "weighted_f1"),
    ("Matthews correlation coefficient (MCC)", "matthews_correlation_coefficient"),
    ("Cohen's kappa", "cohen_kappa"),
    ("Non-drowsy precision / NPV", "non_drowsy_precision"),
    ("Non-drowsy recall / specificity", "non_drowsy_recall_specificity"),
    ("Non-drowsy F1 score", "non_drowsy_f1"),
    ("ROC-AUC", "roc_auc"),
    ("Average precision (AP)", "average_precision"),
    ("False-positive rate", "false_positive_rate"),
    ("False-negative rate", "false_negative_rate"),
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _percent(value: Any) -> str:
    return "Not recorded" if value is None else f"{float(value):.6f} ({float(value) * 100:.4f}%)"


def _metric_rows(metrics: dict[str, Any]) -> list[list[str]]:
    return [[name, _percent(metrics.get(key))] for name, key in METRICS]


def _findings(metrics: dict[str, Any]) -> list[str]:
    matrix = metrics["confusion_matrix_label_order_0_1"]
    true_drowsy, missed_drowsy = matrix[0]
    false_drowsy, true_non_drowsy = matrix[1]
    return [
        f"The official test set contained {metrics['sample_count']:,} images: "
        f"{metrics['drowsy_count']:,} drowsy and {metrics['non_drowsy_count']:,} non-drowsy.",
        f"The model correctly classified {true_drowsy:,} drowsy and "
        f"{true_non_drowsy:,} non-drowsy images.",
        f"There were {missed_drowsy:,} missed drowsy cases (false negatives) and "
        f"{false_drowsy:,} false drowsy alarms (false positives).",
        f"The decision threshold was {metrics['threshold']:.9f}, selected on validation data "
        "by maximizing drowsy-class F2 before the official test evaluation.",
        "The near-perfect result must be interpreted cautiously: the dataset has no driver "
        "identifiers, so subject-independent generalization has not been demonstrated.",
    ]


def _error_summary(metrics: dict[str, Any]) -> str:
    matrix = metrics["confusion_matrix_label_order_0_1"]
    return f"It made {matrix[0][1]:,} false negative(s) and {matrix[1][0]:,} false positive(s)."


def _add_docx_table(document: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Light Shading Accent 1"
    for index, header in enumerate(headers):
        table.rows[0].cells[index].text = header
    for row in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            cells[index].text = value


def write_docx(path: Path, metrics: dict[str, Any], config: Any, audit: dict[str, Any]) -> None:
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    styles = document.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"].font.size = Pt(10)
    title = document.add_heading("Driver Drowsiness Detection Model", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = document.add_paragraph("Training, evaluation, deployment, and findings report")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph(
        f"Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} | "
        f"Checkpoint epoch {metrics['checkpoint_epoch']}"
    ).alignment = WD_ALIGN_PARAGRAPH.CENTER

    document.add_heading("Executive summary", level=1)
    document.add_paragraph(
        f"MobileNetV3-Small achieved {_percent(metrics['accuracy'])} accuracy and "
        f"{_percent(metrics['matthews_correlation_coefficient'])} MCC on the untouched "
        f"official test split ({metrics['sample_count']:,} images). {_error_summary(metrics)} "
        "The model is deployed as FP32 ONNX "
        "for local browser inference."
    )
    document.add_heading("Model and data", level=1)
    _add_docx_table(
        document,
        ["Item", "Recorded value"],
        [
            ["Architecture", config.model.architecture],
            [
                "Input",
                f"RGB, {config.model.image_size} x {config.model.image_size}, "
                "ImageNet normalization",
            ],
            ["Dataset", config.data.dataset_id],
            ["Pinned revision", metrics["dataset_revision"]],
            ["Positive class", "Drowsy (label 0)"],
            ["Training epochs", str(metrics["checkpoint_epoch"])],
            ["Training device", metrics["device"]],
            [
                "Audit status",
                "Full audit artifact available" if audit else "Audit artifact unavailable",
            ],
        ],
    )
    document.add_heading("Official test metrics", level=1)
    _add_docx_table(document, ["Metric", "Value"], _metric_rows(metrics))
    document.add_heading("Confusion matrix", level=1)
    matrix = metrics["confusion_matrix_label_order_0_1"]
    _add_docx_table(
        document,
        ["Actual / predicted", "Drowsy", "Non Drowsy"],
        [
            ["Drowsy", str(matrix[0][0]), str(matrix[0][1])],
            ["Non Drowsy", str(matrix[1][0]), str(matrix[1][1])],
        ],
    )
    document.add_heading("Findings and interpretation", level=1)
    for finding in _findings(metrics):
        document.add_paragraph(finding, style="List Bullet")
    document.add_heading("Deployment and how to try it", level=1)
    document.add_paragraph(
        "The trained PyTorch checkpoint is exported to ONNX with a fixed [1, 3, 224, 224] "
        "float32 input. The browser detects the largest face, applies a padded crop and ImageNet "
        "normalization, then runs the model locally using WebGPU with a WASM fallback. Run "
        "`make web-dev`, open http://127.0.0.1:5173, and choose Start camera or Test an image."
    )
    document.add_heading("Limitations and safety", level=1)
    document.add_paragraph(
        "This is an experimental thesis prototype, not a certified automotive safety system. "
        "Do not test it while operating a vehicle. Dataset subject IDs are unavailable; exact "
        "duplicate controls do not substitute for a driver-disjoint evaluation. Real-world "
        "lighting, eyewear, pose, camera quality, demographics, and temporal behavior require "
        "additional testing. A single-image result is a model score, not a medical diagnosis."
    )
    document.save(path)


def write_pdf(path: Path, metrics: dict[str, Any], config: Any) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CenteredTitle", parent=styles["Title"], alignment=TA_CENTER))
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
    )
    story: list[Any] = [
        Paragraph("Driver Drowsiness Detection Model", styles["CenteredTitle"]),
        Paragraph("Training, evaluation, deployment, and findings report", styles["Heading2"]),
        Spacer(1, 4 * mm),
        Paragraph("Executive summary", styles["Heading1"]),
        Paragraph(
            f"MobileNetV3-Small achieved {_percent(metrics['accuracy'])} accuracy and "
            f"{_percent(metrics['matthews_correlation_coefficient'])} MCC on "
            f"{metrics['sample_count']:,} official test images. The confusion matrix contains "
            f"the following errors: {_error_summary(metrics)}",
            styles["BodyText"],
        ),
        Spacer(1, 3 * mm),
        Paragraph("Model and data", styles["Heading1"]),
    ]
    summary_rows = [
        ["Item", "Recorded value"],
        ["Architecture", config.model.architecture],
        ["Input", f"RGB {config.model.image_size} x {config.model.image_size}"],
        ["Dataset", config.data.dataset_id],
        ["Revision", metrics["dataset_revision"]],
        ["Positive class", "Drowsy (label 0)"],
        ["Checkpoint", f"Epoch {metrics['checkpoint_epoch']} on {metrics['device']}"],
    ]
    story.append(_pdf_table(summary_rows, [43 * mm, 120 * mm]))
    story.extend([Spacer(1, 3 * mm), Paragraph("Official test metrics", styles["Heading1"])])
    story.append(_pdf_table([["Metric", "Value"], *_metric_rows(metrics)], [95 * mm, 68 * mm]))
    story.extend([PageBreak(), Paragraph("Confusion matrix", styles["Heading1"])])
    matrix = metrics["confusion_matrix_label_order_0_1"]
    story.append(
        _pdf_table(
            [
                ["Actual / predicted", "Drowsy", "Non Drowsy"],
                ["Drowsy", str(matrix[0][0]), str(matrix[0][1])],
                ["Non Drowsy", str(matrix[1][0]), str(matrix[1][1])],
            ],
            [65 * mm, 49 * mm, 49 * mm],
        )
    )
    story.extend([Spacer(1, 3 * mm), Paragraph("Findings and interpretation", styles["Heading1"])])
    for finding in _findings(metrics):
        story.append(Paragraph(f"• {finding}", styles["BodyText"]))
        story.append(Spacer(1, 1.5 * mm))
    story.extend(
        [
            Paragraph("How to try it", styles["Heading1"]),
            Paragraph(
                "Run <b>make web-dev</b>, open <b>http://127.0.0.1:5173</b>, and select "
                "<b>Start camera</b> or <b>Test an image</b>. Inference remains in the browser.",
                styles["BodyText"],
            ),
            Paragraph("Limitations and safety", styles["Heading1"]),
            Paragraph(
                "Experimental thesis prototype only; not a certified automotive safety system. "
                "Never test while driving. Subject-independent generalization has not been "
                "established, and "
                "real-world conditions require further validation.",
                styles["BodyText"],
            ),
        ]
    )
    doc.build(story)


def _pdf_table(rows: list[list[str]], widths: list[float]) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16324F")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#A8B3BF")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF3F7")]),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--metrics", default="artifacts/evaluation/test-metrics.json")
    parser.add_argument("--audit", default="artifacts/audit/dataset-audit.json")
    parser.add_argument("--output-dir", default="artifacts/reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    metrics = _load_json(args.metrics)
    audit_path = Path(args.audit)
    audit = _load_json(audit_path) if audit_path.exists() else {}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    docx_path = output_dir / "driver-drowsiness-model-report.docx"
    pdf_path = output_dir / "driver-drowsiness-model-report.pdf"
    write_docx(docx_path, metrics, config, audit)
    write_pdf(pdf_path, metrics, config)
    print(f"DOCX: {docx_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
