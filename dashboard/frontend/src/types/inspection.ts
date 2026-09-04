export type Decision = "pass" | "review" | "fail" | "PASS" | "REVIEW" | "FAIL" | "REQUEST_RECAPTURE";

export type DefectLabel = "normal" | "scratch" | "dent" | "paint_defect" | "paint_misalignment" | "unknown" | "Scratch" | "Dent" | "Paint Defect" | "Paint Misalignment" | "Good";

export interface EvidenceRegion {
  x: number;
  y: number;
  width: number;
  height: number;
  label?: string;
}

export interface Evidence {
  perspective: "appearance" | "texture" | "geometry" | "context" | "uncertainty" | "novelty" | string;
  score: number | null;
  confidence: number | null;
  status: "available" | "unavailable" | "error" | "AVAILABLE" | "UNAVAILABLE" | "SKIPPED" | "CONTRADICTORY";
  message?: string;
  region?: number[];
  regions?: EvidenceRegion[];
  reference?: string;
  runtime_ms?: number;
  metadata?: Record<string, any>;
}

export interface ImageRecord {
  image_id: string;
  file_path: string;
  timestamp: string;
  camera_id: string;
  part_id?: string;
  view_id?: string;
}

export interface Prediction {
  label: string;
  confidence: number;
  bbox?: number[];
}

export interface ModelMetadata {
  version: string;
  name: string;
  thresholds: Record<string, number>;
}

export interface InspectionResult {
  inspection_id: string;
  image_url?: string;
  decision: Decision;
  predicted_label?: DefectLabel;
  primary_prediction?: Prediction;
  prediction?: Prediction;
  confidence?: number;
  anomaly_score?: number | null;
  severity: "none" | "low" | "medium" | "high" | "critical" | "unknown" | string;
  reason?: string;
  rule_fired?: string;
  reasoning_chain?: string[];
  evidence: Evidence[];
  model_version?: string;
  configuration_version?: string;
  calibration_version?: string;
  model_metadata?: ModelMetadata;
  image_record?: ImageRecord;
  timings_ms?: Record<string, number>;
  latency_ms?: number;
  created_at?: string;
  timestamp?: string;
}

export interface DatasetAuditFinding {
  category: string;
  severity: string;
  message: string;
  affected_files: string[];
}

export interface DatasetAuditReport {
  dataset_root: string;
  total_images: number;
  valid_images: number;
  corrupted_images: number;
  duplicate_images: number;
  split_leakage_count: number;
  class_counts: Record<string, number>;
  split_counts: Record<string, number>;
  dimension_stats: Record<string, any>;
  findings: DatasetAuditFinding[];
  passed_quality_gate: boolean;
}

export interface FeedbackSubmission {
  inspection_id: string;
  corrected_decision: Decision;
  corrected_label?: string;
  notes?: string;
}
