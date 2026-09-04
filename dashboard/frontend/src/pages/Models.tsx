import React, { useState } from 'react';

export const ModelsPage: React.FC = () => {
  const [showConfirm, setShowConfirm] = useState<boolean>(false);

  return (
    <div className="space-y-6">
      <div className="bg-white p-4 rounded-lg border border-[#dbe2ea] shadow-sm flex justify-between items-center">
        <div>
          <h1 className="text-xl font-bold text-slate-900">Model Registry & Version Management</h1>
          <p className="text-xs text-slate-500 mt-0.5">Multi-head neural backbones, checkpoint lineage, and ONNX deployment status</p>
        </div>
        <button
          onClick={() => setShowConfirm(true)}
          className="bg-[#2457d6] hover:bg-blue-700 text-white text-xs font-bold px-3 py-1.5 rounded shadow-sm"
        >
          Promote Checkpoint v2.0.0
        </button>
      </div>

      {showConfirm && (
        <div className="bg-amber-50 border border-amber-300 p-4 rounded-lg text-xs space-y-2">
          <div className="font-bold text-amber-900">Confirm Production Deployment</div>
          <p className="text-amber-800">
            Promoting EfficientNet_B0_MultiHead v2.0.0 will update default inference weights across edge inspection nodes.
          </p>
          <div className="flex gap-2 pt-1">
            <button onClick={() => setShowConfirm(false)} className="bg-amber-800 text-white px-3 py-1 rounded font-bold">
              Confirm Release
            </button>
            <button onClick={() => setShowConfirm(false)} className="bg-white border border-amber-300 px-3 py-1 rounded">
              Cancel
            </button>
          </div>
        </div>
      )}

      <div className="bg-white border border-[#dbe2ea] rounded-lg shadow-sm overflow-hidden">
        <table className="w-full text-left text-xs font-sans">
          <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-bold uppercase tracking-wider text-[11px]">
            <tr>
              <th className="p-3">Version</th>
              <th className="p-3">Architecture</th>
              <th className="p-3">Macro F1</th>
              <th className="p-3">ONNX Status</th>
              <th className="p-3">Release State</th>
              <th className="p-3">Created</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 font-mono">
            <tr className="bg-blue-50/40">
              <td className="p-3 font-bold text-blue-900">v2.0.0 (Active)</td>
              <td className="p-3 font-sans">EfficientNet-B0 MultiHead</td>
              <td className="p-3 font-bold text-emerald-700">0.9418</td>
              <td className="p-3 text-emerald-700 font-sans font-semibold">✓ FP32 Exported (19.8 MB)</td>
              <td className="p-3 font-sans"><span className="bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded text-[11px] font-bold">PRODUCTION</span></td>
              <td className="p-3 text-slate-500 font-sans">2026-08-30</td>
            </tr>
            <tr>
              <td className="p-3 font-medium text-slate-700">v1.0.0</td>
              <td className="p-3 font-sans">RGB Baseline Feature Extractor</td>
              <td className="p-3 text-slate-700">0.8950</td>
              <td className="p-3 text-slate-500 font-sans">Fallback Engine</td>
              <td className="p-3 font-sans"><span className="bg-slate-100 text-slate-600 px-2 py-0.5 rounded text-[11px]">ARCHIVED</span></td>
              <td className="p-3 text-slate-500 font-sans">2026-08-15</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};
