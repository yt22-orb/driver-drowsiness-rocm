export class ScoreWindow {
  private values: Array<{ time: number; score: number }> = [];
  private warningStartedAt: number | null = null;
  private clearStartedAt: number | null = null;
  public warning = false;

  constructor(
    private readonly windowSeconds: number,
    private readonly minimumValidFrames: number,
    private readonly threshold: number,
    private readonly clearMargin: number,
    private readonly clearSeconds: number,
  ) {}

  add(score: number, nowMs: number): { average: number; warning: boolean; possible: boolean } {
    const windowMs = this.windowSeconds * 1000;
    this.values.push({ time: nowMs, score });
    this.values = this.values.filter((item) => nowMs - item.time <= windowMs);
    const average = this.values.reduce((sum, item) => sum + item.score, 0) / this.values.length;
    const coversWindow =
      this.values.length >= this.minimumValidFrames &&
      nowMs - (this.values[0]?.time ?? nowMs) >= windowMs * 0.85;

    if (average >= this.threshold && coversWindow) {
      this.warningStartedAt ??= nowMs;
      this.clearStartedAt = null;
      if (nowMs - this.warningStartedAt >= windowMs * 0.15) this.warning = true;
    } else {
      this.warningStartedAt = null;
      if (this.warning && average < Math.max(0, this.threshold - this.clearMargin)) {
        this.clearStartedAt ??= nowMs;
        if (nowMs - this.clearStartedAt >= this.clearSeconds * 1000) this.warning = false;
      } else if (!this.warning) {
        this.clearStartedAt = null;
      }
    }
    return { average, warning: this.warning, possible: average >= this.threshold };
  }

  reset(): void {
    this.values = [];
    this.warningStartedAt = null;
    this.clearStartedAt = null;
    this.warning = false;
  }
}
