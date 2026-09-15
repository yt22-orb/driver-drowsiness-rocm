export type ModelArchitecture = "drowsiness_cnn_v1" | "mobilenet_v3_small";

export interface ModelMetadata {
  schemaVersion: 1 | 2;
  modelVersion: string;
  modelFile: string;
  architecture?: ModelArchitecture;
  architectureDetails?: {
    name: "drowsiness_cnn_v1";
    family: string;
    parameterCount: number;
    initialization: string;
    stem: string;
    stages: Array<{ channels: number; blocks: number; firstStride: number }>;
    head: string;
  };
  weights?: {
    origin: string;
    externalCheckpoint: false;
    checkpointEpoch: number;
  };
  labels: [string, string];
  drowsyLabelIndex: number;
  input: {
    name: string;
    layout: "NCHW";
    dtype: "float32";
    width: number;
    height: number;
    colorSpace?: "RGB";
    valueRangeBeforeNormalization?: [number, number];
    mean: [number, number, number];
    std: [number, number, number];
    source?: "centered square face crop";
    cropStrategy?: "center-square";
    cropFraction?: number;
    interpolation?: string;
  };
  output: { name: string; shape: [number, number] };
  decision: {
    drowsyThreshold: number;
    beta: number;
    windowSeconds: number;
    minimumValidFrames: number;
    clearMargin: number;
    clearSeconds: number;
  };
  warning: string;
}

export type AppState =
  | "loading"
  | "ready"
  | "positioning"
  | "no-face"
  | "attentive"
  | "possible"
  | "warning"
  | "error";
