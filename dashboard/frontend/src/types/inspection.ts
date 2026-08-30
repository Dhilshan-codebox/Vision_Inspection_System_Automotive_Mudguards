export interface ImageRecord {
  image_id: string;
  file_path: string;
  timestamp: string;
  camera_id: string;
  part_id?: string;
  view_id?: string;
}

export interface Evidence {
  perspective: string;
  status: 'AVAILABLE' | 'UNAVAILABLE' | 'SKIPPED' | 'CONTRADICTORY';
  confidence: number;
  score?: number;
  region?: number[];
  reference?: string;
  runtime_ms: number;
  metadata: Record<string, any>;
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
  decision: 'PASS' | 'REVIEW' | 'FAIL' | 'REQUEST_RECAPTURE';
  image_record: ImageRecord;
  primary_prediction?: Prediction;
  evidence: Evidence[];
  severity: string;
  model_metadata?: ModelMetadata;
  timestamp: string;
  latency_ms: number;
  rule_fired?: string;
  reasoning_chain: string[];
}
