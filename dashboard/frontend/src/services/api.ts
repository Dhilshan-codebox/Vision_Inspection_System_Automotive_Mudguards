import { InspectionResult, DatasetAuditReport, FeedbackSubmission } from '../types/inspection';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export class ApiClient {
  static async checkHealth(): Promise<{ status: string; version: string }> {
    try {
      const res = await fetch(`${API_BASE}/`);
      if (!res.ok) throw new Error('Health check failed');
      return await res.json();
    } catch {
      return { status: 'SIMULATION', version: '2.0.0-mock' };
    }
  }

  static async inspectImage(file: File, cameraId: string = 'cam_top', partId?: string): Promise<InspectionResult> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('camera_id', cameraId);
    if (partId) formData.append('part_id', partId);

    try {
      const res = await fetch(`${API_BASE}/api/v1/inspect`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error('Inspection API error');
      return await res.json();
    } catch (e) {
      // Software-only simulation fallback mode
      return {
        inspection_id: `ins_sim_${Math.random().toString(36).substring(2, 9)}`,
        decision: 'FAIL',
        predicted_label: 'scratch',
        primary_prediction: {
          label: 'Scratch',
          confidence: 0.88,
          bbox: [150, 300, 500, 322],
        },
        severity: 'medium',
        reason: 'High texture roughness 0.58 confirmed with nominal context',
        rule_fired: 'RULE_5_TEXTURE_CONFIRMED_DEFECT',
        reasoning_chain: [
          'High texture roughness 0.58 confirmed with nominal context',
          'Appearance perspective confirmed sharp gradient discontinuity',
        ],
        evidence: [
          {
            perspective: 'appearance',
            status: 'AVAILABLE',
            confidence: 0.88,
            score: 0.65,
            region: [150, 300, 500, 322],
            reference: 'multiscale_gradient_appearance_map',
            runtime_ms: 12.4,
            metadata: { defect_type_hint: 'scratch_candidate' },
          },
          {
            perspective: 'texture',
            status: 'AVAILABLE',
            confidence: 0.88,
            score: 0.58,
            runtime_ms: 45.1,
            metadata: { mean_roughness: 28.5 },
          },
          {
            perspective: 'geometry',
            status: 'UNAVAILABLE',
            confidence: 0.0,
            score: null,
            message: 'Geometry unavailable: No 3D/depth sensor connected',
            runtime_ms: 0.0,
            metadata: { reason: 'No 3D depth map attached' },
          },
          {
            perspective: 'uncertainty',
            status: 'AVAILABLE',
            confidence: 0.90,
            score: 0.15,
            runtime_ms: 2.1,
            metadata: { uncertainty_type: 'nominal' },
          },
        ],
        model_version: '2.0.0',
        configuration_version: '1.0.0',
        calibration_version: '1.0.0',
        latency_ms: 59.6,
        created_at: new Date().toISOString(),
      };
    }
  }

  static async submitFeedback(feedback: FeedbackSubmission): Promise<{ status: string }> {
    try {
      const res = await fetch(`${API_BASE}/api/v1/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(feedback),
      });
      return await res.json();
    } catch {
      return { status: 'recorded_mock_queue' };
    }
  }
}
