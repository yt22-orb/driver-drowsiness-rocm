import { FaceDetector, FilesetResolver, type Detection } from "@mediapipe/tasks-vision";
import * as ort from "onnxruntime-web/webgpu";
import "./style.css";
import { ScoreWindow } from "./score-window";
import type { AppState, ModelMetadata } from "./types";

const byId = <T extends HTMLElement>(id: string): T => {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing element #${id}`);
  return element as T;
};

const video = byId<HTMLVideoElement>("video");
const uploadedImage = byId<HTMLImageElement>("uploaded-image");
const overlay = byId<HTMLCanvasElement>("overlay");
const emptyState = byId<HTMLDivElement>("empty-state");
const statusElement = byId<HTMLElement>("status");
const stateDot = byId<HTMLSpanElement>("state-dot");
const scoreElement = byId<HTMLElement>("score");
const meterFill = byId<HTMLDivElement>("meter-fill");
const runtimeElement = byId<HTMLElement>("runtime");
const alertElement = byId<HTMLElement>("alert");
const startButton = byId<HTMLButtonElement>("start-camera");
const stopButton = byId<HTMLButtonElement>("stop-camera");
const imageInput = byId<HTMLInputElement>("image-input");
const muteButton = byId<HTMLButtonElement>("mute");

let metadata: ModelMetadata;
let session: ort.InferenceSession;
let videoDetector: FaceDetector;
let imageDetector: FaceDetector;
let scoreWindow: ScoreWindow;
let stream: MediaStream | null = null;
let animationFrame = 0;
let lastInferenceAt = 0;
let muted = false;
let alarmTimer: number | null = null;
let audioContext: AudioContext | null = null;
let currentState: AppState = "loading";
let faceDelegate = "GPU";

const stateText: Record<AppState, string> = {
  loading: "Loading model…",
  ready: "Ready",
  "no-face": "No face detected",
  attentive: "No drowsiness detected",
  possible: "Checking sustained signal…",
  warning: "Possible drowsiness detected",
  error: "Model unavailable",
};

function setState(state: AppState, message?: string): void {
  currentState = state;
  statusElement.textContent = message ?? stateText[state];
  stateDot.dataset.state = state;
  alertElement.hidden = state !== "warning";
  if (state === "warning") startAlarm();
  else stopAlarm();
}

function setScore(score: number | null): void {
  if (score === null) {
    scoreElement.textContent = "—";
    meterFill.style.width = "0%";
    return;
  }
  const percent = Math.round(score * 100);
  scoreElement.textContent = `${percent}%`;
  meterFill.style.width = `${percent}%`;
}

async function initialize(): Promise<void> {
  try {
    const metadataResponse = await fetch("/models/model-metadata.json", { cache: "no-store" });
    const metadataType = metadataResponse.headers.get("content-type") ?? "";
    if (!metadataResponse.ok || !metadataType.includes("application/json")) {
      throw new Error("Trained model metadata is missing. Run training, evaluation, and export first.");
    }
    metadata = (await metadataResponse.json()) as ModelMetadata;
    if (metadata.schemaVersion !== 1 || metadata.labels.length !== 2) {
      throw new Error("Unsupported model metadata contract");
    }

    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";
    const prefersWebGpu = "gpu" in navigator;
    const executionProviders = prefersWebGpu ? ["webgpu", "wasm"] : ["wasm"];
    session = await ort.InferenceSession.create(`/models/${metadata.modelFile}`, {
      executionProviders,
      graphOptimizationLevel: "all",
    });
    runtimeElement.textContent = `Inference: ${prefersWebGpu ? "WebGPU with WASM fallback" : "WASM CPU"}`;

    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm",
    );
    const detectorModel =
      "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite";
    try {
      videoDetector = await FaceDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: detectorModel, delegate: "GPU" },
        runningMode: "VIDEO",
        minDetectionConfidence: 0.5,
      });
      imageDetector = await FaceDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: detectorModel, delegate: "GPU" },
        runningMode: "IMAGE",
        minDetectionConfidence: 0.5,
      });
    } catch (gpuError) {
      console.warn("MediaPipe GPU delegate unavailable; falling back to CPU", gpuError);
      faceDelegate = "CPU";
      videoDetector = await FaceDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: detectorModel, delegate: "CPU" },
        runningMode: "VIDEO",
        minDetectionConfidence: 0.5,
      });
      imageDetector = await FaceDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: detectorModel, delegate: "CPU" },
        runningMode: "IMAGE",
        minDetectionConfidence: 0.5,
      });
    }
    runtimeElement.textContent += `; face detector: ${faceDelegate}`;
    scoreWindow = new ScoreWindow(
      metadata.decision.windowSeconds,
      metadata.decision.minimumValidFrames,
      metadata.decision.drowsyThreshold,
      metadata.decision.clearMargin,
      metadata.decision.clearSeconds,
    );
    startButton.disabled = false;
    imageInput.disabled = false;
    setState("ready");
  } catch (error) {
    console.error(error);
    setState("error", error instanceof Error ? error.message : "Model initialization failed");
    runtimeElement.textContent = "See README: the trained ONNX model is not committed to GitHub.";
  }
}

function largestFace(detections: Detection[]): Detection | null {
  return (
    detections.reduce<Detection | null>((largest, detection) => {
      const box = detection.boundingBox;
      const largestBox = largest?.boundingBox;
      if (!box) return largest;
      if (!largestBox || box.width * box.height > largestBox.width * largestBox.height) {
        return detection;
      }
      return largest;
    }, null) ?? null
  );
}

function faceTensor(
  source: HTMLVideoElement | HTMLImageElement,
  detection: Detection,
): ort.Tensor | null {
  const box = detection.boundingBox;
  if (!box) return null;
  const sourceWidth = source instanceof HTMLVideoElement ? source.videoWidth : source.naturalWidth;
  const sourceHeight = source instanceof HTMLVideoElement ? source.videoHeight : source.naturalHeight;
  const paddingX = box.width * 0.15;
  const paddingY = box.height * 0.15;
  const x = Math.max(0, box.originX - paddingX);
  const y = Math.max(0, box.originY - paddingY);
  const width = Math.min(sourceWidth - x, box.width + paddingX * 2);
  const height = Math.min(sourceHeight - y, box.height + paddingY * 2);
  if (width <= 0 || height <= 0) return null;

  const canvas = document.createElement("canvas");
  canvas.width = metadata.input.width;
  canvas.height = metadata.input.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return null;
  context.drawImage(source, x, y, width, height, 0, 0, canvas.width, canvas.height);
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  const plane = canvas.width * canvas.height;
  const values = new Float32Array(plane * 3);
  for (let pixelIndex = 0; pixelIndex < plane; pixelIndex += 1) {
    for (let channel = 0; channel < 3; channel += 1) {
      const raw = pixels[pixelIndex * 4 + channel] ?? 0;
      values[channel * plane + pixelIndex] =
        (raw / 255 - metadata.input.mean[channel]!) / metadata.input.std[channel]!;
    }
  }
  return new ort.Tensor("float32", values, [1, 3, canvas.height, canvas.width]);
}

async function classify(
  source: HTMLVideoElement | HTMLImageElement,
  detection: Detection,
): Promise<number | null> {
  const tensor = faceTensor(source, detection);
  if (!tensor) return null;
  const results = await session.run({ [metadata.input.name]: tensor });
  const logits = results[metadata.output.name]?.data;
  if (!logits || logits.length !== 2) throw new Error("Unexpected classifier output");
  const first = Number(logits[0]);
  const second = Number(logits[1]);
  const maximum = Math.max(first, second);
  const drowsy = Math.exp(first - maximum);
  const nonDrowsy = Math.exp(second - maximum);
  return drowsy / (drowsy + nonDrowsy);
}

function drawFace(detection: Detection | null, sourceWidth: number, sourceHeight: number): void {
  overlay.width = sourceWidth;
  overlay.height = sourceHeight;
  const context = overlay.getContext("2d");
  if (!context || !detection?.boundingBox) return;
  const box = detection.boundingBox;
  context.strokeStyle = currentState === "warning" ? "#ff4e57" : "#58d6b2";
  context.lineWidth = Math.max(2, sourceWidth / 240);
  context.strokeRect(box.originX, box.originY, box.width, box.height);
}

async function processVideoFrame(now: number): Promise<void> {
  animationFrame = requestAnimationFrame((timestamp) => void processVideoFrame(timestamp));
  if (!stream || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || now - lastInferenceAt < 100) {
    return;
  }
  lastInferenceAt = now;
  try {
    const result = videoDetector.detectForVideo(video, now);
    const face = largestFace(result.detections);
    drawFace(face, video.videoWidth, video.videoHeight);
    if (!face) {
      scoreWindow.reset();
      setScore(null);
      setState("no-face");
      return;
    }
    const score = await classify(video, face);
    if (score === null) return;
    const smoothed = scoreWindow.add(score, now);
    setScore(smoothed.average);
    setState(smoothed.warning ? "warning" : smoothed.possible ? "possible" : "attentive");
  } catch (error) {
    console.error(error);
    stopCamera();
    setState("error", "Inference failed; check the browser console");
  }
}

async function startCamera(): Promise<void> {
  try {
    audioContext ??= new AudioContext();
    await audioContext.resume();
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    uploadedImage.hidden = true;
    video.hidden = false;
    emptyState.hidden = true;
    startButton.disabled = true;
    stopButton.disabled = false;
    scoreWindow.reset();
    setState("no-face");
    animationFrame = requestAnimationFrame((timestamp) => void processVideoFrame(timestamp));
  } catch (error) {
    console.error(error);
    setState("error", "Camera permission was denied or no camera is available");
  }
}

function stopCamera(): void {
  cancelAnimationFrame(animationFrame);
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  overlay.getContext("2d")?.clearRect(0, 0, overlay.width, overlay.height);
  scoreWindow?.reset();
  stopAlarm();
  startButton.disabled = false;
  stopButton.disabled = true;
  setScore(null);
  setState("ready");
}

async function processUploadedImage(file: File): Promise<void> {
  if (!['image/jpeg', 'image/png'].includes(file.type)) {
    setState("error", "Only JPEG and PNG images are supported");
    return;
  }
  stopCamera();
  const objectUrl = URL.createObjectURL(file);
  try {
    uploadedImage.src = objectUrl;
    await uploadedImage.decode();
    video.hidden = true;
    uploadedImage.hidden = false;
    emptyState.hidden = true;
    const result = imageDetector.detect(uploadedImage);
    const face = largestFace(result.detections);
    drawFace(face, uploadedImage.naturalWidth, uploadedImage.naturalHeight);
    if (!face) {
      setScore(null);
      setState("no-face");
      return;
    }
    const score = await classify(uploadedImage, face);
    if (score === null) return;
    setScore(score);
    setState(score >= metadata.decision.drowsyThreshold ? "possible" : "attentive");
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

function beep(): void {
  if (muted || !audioContext) return;
  const oscillator = audioContext.createOscillator();
  const gain = audioContext.createGain();
  oscillator.frequency.value = 880;
  oscillator.type = "square";
  gain.gain.setValueAtTime(0.0001, audioContext.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.18, audioContext.currentTime + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.0001, audioContext.currentTime + 0.25);
  oscillator.connect(gain).connect(audioContext.destination);
  oscillator.start();
  oscillator.stop(audioContext.currentTime + 0.26);
}

function startAlarm(): void {
  if (alarmTimer !== null || muted) return;
  beep();
  alarmTimer = window.setInterval(beep, 800);
}

function stopAlarm(): void {
  if (alarmTimer !== null) window.clearInterval(alarmTimer);
  alarmTimer = null;
}

startButton.addEventListener("click", () => void startCamera());
stopButton.addEventListener("click", stopCamera);
imageInput.addEventListener("change", () => {
  const file = imageInput.files?.[0];
  if (file) void processUploadedImage(file);
});
muteButton.addEventListener("click", () => {
  muted = !muted;
  muteButton.textContent = `Sound: ${muted ? "off" : "on"}`;
  muteButton.setAttribute("aria-pressed", String(muted));
  if (muted) stopAlarm();
  else if (currentState === "warning") startAlarm();
});
window.addEventListener("pagehide", stopCamera);

void initialize();
