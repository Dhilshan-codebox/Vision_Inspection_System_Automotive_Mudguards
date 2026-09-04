import React, { useState } from 'react';
import { DecisionBadge } from '../components/DecisionBadge';
import { EvidenceViewer } from '../components/EvidenceViewer';
import { ReasoningTrace } from '../components/ReasoningTrace';
import { InspectionResult } from '../types/inspection';
import { ApiClient } from '../services/api';

const MOCK_RESULT: InspectionResult = {
  inspection_id: 'ins_mock_static_001',
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
  model_metadata: {
    version: '2.0.0',
    name: 'EfficientNet_B0_MultiHead',
    thresholds: { fail_defect_threshold: 0.6 },
  },
  image_record: {
    image_id: 'img_sample_scratch',
    file_path: 'data/train/Scratch/part01_camTop.jpg',
    timestamp: new Date().toISOString(),
    camera_id: 'cam_top',
    part_id: 'mudguard_front_left',
  },
  model_version: '2.0.0',
  configuration_version: '1.0.0',
  calibration_version: '1.0.0',
  latency_ms: 59.6,
  created_at: new Date().toISOString(),
};

export const InspectionPage: React.FC = () => {
  const [result, setResult] = useState<InspectionResult>(MOCK_RESULT);
  const [previewUrl, setPreviewUrl] = useState<string | undefined>();
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [feedbackNote, setFeedbackNote] = useState<string>('');
  const [feedbackSent, setFeedbackSent] = useState<boolean>(false);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setPreviewUrl(URL.createObjectURL(file));
      setIsUploading(true);
      setFeedbackSent(false);

      const inspectRes = await ApiClient.inspectImage(file, 'cam_top', 'mudguard_part_001');
      setResult(inspectRes);
      setIsUploading(false);
    }
  };

  const handleFeedback = async (correctedDecision: 'PASS' | 'REVIEW' | 'FAIL') => {
    await ApiClient.submitFeedback({
      inspection_id: result.inspection_id,
      corrected_decision: correctedDecision,
      notes: feedbackNote,
    });
    setFeedbackSent(true);
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center bg-white p-4 rounded-lg border border-[#dbe2ea] shadow-sm">
        <div>
          <h1 className="text-xl font-bold text-slate-900">Operator Inspection Workspace</h1>
          <p className="text-xs text-slate-500 mt-0.5">Software-first visual quality analysis for automotive mudguards</p>
        </div>
        <DecisionBadge decision={result.decision} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          <div className="bg-white border border-[#dbe2ea] p-4 rounded-lg flex items-center justify-between shadow-sm">
            <span className="text-xs font-semibold text-slate-700">Upload Frame or Select Preset Sample:</span>
            <input
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              className="text-xs text-slate-600 file:mr-4 file:py-1.5 file:px-3 file:rounded file:border-0 file:text-xs file:font-semibold file:bg-[#2457d6] file:text-white hover:file:bg-blue-700 cursor-pointer"
            />
          </div>

          {isUploading && (
            <div className="bg-blue-50 border border-blue-200 text-blue-800 text-xs p-3 rounded font-medium">
              Executing multi-perspective inference pipeline...
            </div>
          )}

          <EvidenceViewer
            imageUrl={previewUrl}
            prediction={result.primary_prediction || result.prediction}
            evidence={result.evidence}
          />
        </div>

        <div className="space-y-4">
          <ReasoningTrace
            inspectionId={result.inspection_id}
            ruleFired={result.rule_fired}
            reasoningChain={result.reasoning_chain}
            modelVersion={result.model_version || result.model_metadata?.version}
            configVersion={result.configuration_version}
            calibrationVersion={result.calibration_version}
          />

          <div className="bg-white border border-[#dbe2ea] p-4 rounded-lg space-y-2 text-xs shadow-sm">
            <div className="font-semibold text-slate-800 border-b border-slate-100 pb-1.5">Inspection Metadata</div>
            <div className="flex justify-between text-slate-600">
              <span>Part ID:</span>
              <span className="font-mono text-slate-900 font-medium">{result.image_record?.part_id || 'N/A'}</span>
            </div>
            <div className="flex justify-between text-slate-600">
              <span>Camera View:</span>
              <span className="font-mono text-slate-900 font-medium">{result.image_record?.camera_id || 'cam_main'}</span>
            </div>
            <div className="flex justify-between text-slate-600">
              <span>Primary Class:</span>
              <span className="font-mono font-bold text-slate-900 capitalize">
                {result.primary_prediction?.label || result.predicted_label || 'Normal'}
              </span>
            </div>
            <div className="flex justify-between text-slate-600">
              <span>Severity Level:</span>
              <span className="font-mono text-slate-900 font-medium capitalize">{result.severity}</span>
            </div>
            <div className="flex justify-between text-slate-600">
              <span>Inference Latency:</span>
              <span className="font-mono text-slate-900 font-medium">
                {result.latency_ms !== undefined ? result.latency_ms.toFixed(1) : 'N/A'} ms
              </span>
            </div>
          </div>

          <div className="bg-white border border-[#dbe2ea] p-4 rounded-lg space-y-3 text-xs shadow-sm">
            <div className="font-semibold text-slate-800 border-b border-slate-100 pb-1.5">Human Operator Feedback</div>
            <p className="text-[#64748b] text-[11px]">Record feedback to add frame to active learning review queue:</p>

            <div className="flex gap-2">
              <button
                onClick={() => handleFeedback('PASS')}
                className="flex-1 bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border border-emerald-300 py-1.5 rounded font-bold"
              >
                Confirm PASS
              </button>
              <button
                onClick={() => handleFeedback('REVIEW')}
                className="flex-1 bg-amber-50 hover:bg-amber-100 text-amber-800 border border-amber-300 py-1.5 rounded font-bold"
              >
                Mark REVIEW
              </button>
              <button
                onClick={() => handleFeedback('FAIL')}
                className="flex-1 bg-rose-50 hover:bg-rose-100 text-rose-800 border border-rose-300 py-1.5 rounded font-bold"
              >
                Confirm FAIL
              </button>
            </div>

            <input
              type="text"
              placeholder="Operator notes (optional)..."
              value={feedbackNote}
              onChange={(e) => setFeedbackNote(e.target.value)}
              className="w-full bg-slate-50 border border-slate-200 p-2 rounded text-slate-800 text-xs"
            />

            {feedbackSent && (
              <div className="text-emerald-700 font-semibold text-[11px] bg-emerald-50 p-2 rounded border border-emerald-200">
                ✓ Feedback recorded for active learning queue.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
