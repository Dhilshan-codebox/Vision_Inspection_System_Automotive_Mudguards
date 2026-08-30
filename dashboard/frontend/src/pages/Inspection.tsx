import React, { useState } from 'react';
import { DecisionBadge } from '../components/DecisionBadge';
import { EvidenceViewer } from '../components/EvidenceViewer';
import { ReasoningTrace } from '../components/ReasoningTrace';
import { InspectionResult } from '../types/inspection';

const MOCK_RESULT: InspectionResult = {
  inspection_id: 'ins_mock_static_001',
  decision: 'FAIL',
  image_record: {
    image_id: 'img_sample_scratch',
    file_path: 'data/train/Scratch/part01_camTop.jpg',
    timestamp: new Date().toISOString(),
    camera_id: 'cam_top',
    part_id: 'mudguard_front_left',
  },
  primary_prediction: {
    label: 'Scratch',
    confidence: 0.88,
    bbox: [150, 300, 500, 322],
  },
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
      perspective: 'uncertainty',
      status: 'AVAILABLE',
      confidence: 0.90,
      score: 0.15,
      runtime_ms: 2.1,
      metadata: { uncertainty_type: 'nominal' },
    },
  ],
  severity: 'medium',
  model_metadata: {
    version: '2.0.0',
    name: 'EfficientNet_B0_MultiHead',
    thresholds: { fail_defect_threshold: 0.6 },
  },
  timestamp: new Date().toISOString(),
  latency_ms: 59.6,
  rule_fired: 'RULE_5_TEXTURE_CONFIRMED_DEFECT',
  reasoning_chain: [
    'High texture roughness 0.58 confirmed with nominal context',
    'Appearance perspective confirmed sharp gradient discontinuity',
  ],
};

export const InspectionPage: React.FC = () => {
  const [result, setResult] = useState<InspectionResult>(MOCK_RESULT);
  const [previewUrl, setPreviewUrl] = useState<string | undefined>();

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setPreviewUrl(URL.createObjectURL(file));
      // In software test mode, simulate inspection execution
      setResult({
        ...MOCK_RESULT,
        inspection_id: `ins_upload_${Math.random().toString(36).substring(2, 8)}`,
        image_record: {
          ...MOCK_RESULT.image_record,
          file_path: file.name,
        },
      });
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6 bg-slate-950 text-slate-100 min-h-screen">
      <header className="flex justify-between items-center border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-2xl font-extrabold tracking-tight">Operator Inspection Workspace</h1>
          <p className="text-xs text-slate-400 mt-1">Multi-perspective mudguard defect detection and evidence reasoning</p>
        </div>
        <DecisionBadge decision={result.decision} />
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          <div className="bg-slate-900 border border-slate-800 p-4 rounded-lg flex items-center justify-between">
            <label className="text-xs font-semibold text-slate-300">Upload Frame for Analysis:</label>
            <input
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              className="text-xs text-slate-400 file:mr-4 file:py-1.5 file:px-3 file:rounded file:border-0 file:text-xs file:font-semibold file:bg-slate-800 file:text-slate-200 hover:file:bg-slate-700 cursor-pointer"
            />
          </div>

          <EvidenceViewer
            imageUrl={previewUrl}
            prediction={result.primary_prediction}
            evidence={result.evidence}
          />
        </div>

        <div className="space-y-4">
          <ReasoningTrace
            inspectionId={result.inspection_id}
            ruleFired={result.rule_fired}
            reasoningChain={result.reasoning_chain}
            modelVersion={result.model_metadata?.version}
          />

          <div className="bg-slate-900 border border-slate-800 p-4 rounded-lg space-y-2 text-xs">
            <div className="font-semibold text-slate-300 border-b border-slate-800 pb-1">Part Metadata</div>
            <div className="flex justify-between text-slate-400">
              <span>Part ID:</span> <span className="text-slate-200 font-mono">{result.image_record.part_id}</span>
            </div>
            <div className="flex justify-between text-slate-400">
              <span>Camera View:</span> <span className="text-slate-200 font-mono">{result.image_record.camera_id}</span>
            </div>
            <div className="flex justify-between text-slate-400">
              <span>Severity Estimate:</span> <span className="text-slate-200 font-mono capitalize">{result.severity}</span>
            </div>
            <div className="flex justify-between text-slate-400">
              <span>Total Latency:</span> <span className="text-slate-200 font-mono">{result.latency_ms.toFixed(1)} ms</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
