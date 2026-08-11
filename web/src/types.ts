export interface ModelMetadata {
  schemaVersion: number;
  modelVersion: string;
  modelFile: string;
  labels: [string, string];
  drowsyLabelIndex: number;
  input: {
    name: string;
    layout: "NCHW";
    dtype: "float32";
    width: number;
    height: number;
    mean: [number, number, number];
    std: [number, number, number];
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
  | "no-face"
  | "attentive"
  | "possible"
  | "warning"
  | "error";
