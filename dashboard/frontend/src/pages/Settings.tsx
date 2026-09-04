import React, { useState } from 'react';

export const SettingsPage: React.FC = () => {
  const [failThreshold, setFailThreshold] = useState<number>(0.60);
  const [uncertaintyThreshold, setUncertaintyThreshold] = useState<number>(0.40);
  const [saved, setSaved] = useState<boolean>(false);

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2500);
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="bg-white p-4 rounded-lg border border-[#dbe2ea] shadow-sm flex justify-between items-center">
        <div>
          <h1 className="text-xl font-bold text-slate-900">System Configuration & Thresholds</h1>
          <p className="text-xs text-slate-500 mt-0.5">Versioned quality gate parameters, decision thresholds, and perspective weights</p>
        </div>
        <button
          onClick={handleSave}
          className="bg-[#2457d6] hover:bg-blue-700 text-white text-xs font-bold px-4 py-1.5 rounded shadow-sm"
        >
          Save Configuration
        </button>
      </div>

      {saved && (
        <div className="bg-emerald-50 border border-emerald-300 text-emerald-800 text-xs p-3 rounded font-semibold">
          ✓ Configuration changes saved successfully (Config Version updated to v1.0.1).
        </div>
      )}

      <div className="bg-white p-5 rounded-lg border border-[#dbe2ea] shadow-sm space-y-5 text-xs">
        <div className="border-b border-slate-100 pb-3 font-bold text-slate-800 uppercase tracking-wider text-[11px]">
          Decision & Fusion Thresholds
        </div>

        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <label className="font-semibold text-slate-700">Defect FAIL Decision Threshold:</label>
            <span className="font-mono text-slate-900 font-bold bg-slate-100 px-2 py-0.5 rounded">{failThreshold.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0.30"
            max="0.90"
            step="0.05"
            value={failThreshold}
            onChange={(e) => setFailThreshold(parseFloat(e.target.value))}
            className="w-full accent-blue-600"
          />
          <p className="text-slate-400 text-[11px]">Minimum probability required to trigger automatic FAIL decision.</p>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <label className="font-semibold text-slate-700">Maximum Acceptable Uncertainty (REVIEW Trigger):</label>
            <span className="font-mono text-slate-900 font-bold bg-slate-100 px-2 py-0.5 rounded">{uncertaintyThreshold.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0.20"
            max="0.60"
            step="0.05"
            value={uncertaintyThreshold}
            onChange={(e) => setUncertaintyThreshold(parseFloat(e.target.value))}
            className="w-full accent-blue-600"
          />
          <p className="text-slate-400 text-[11px]">Uncertainty score exceeding this threshold triggers operator review.</p>
        </div>

        <div className="border-t border-slate-100 pt-4 space-y-2">
          <div className="font-semibold text-slate-800">Perspective Module Weights</div>
          <div className="grid grid-cols-2 gap-3 text-slate-600 font-mono text-[11px]">
            <div className="p-2 bg-slate-50 rounded border border-slate-100 flex justify-between">
              <span>RGB Backbone:</span> <span>0.35</span>
            </div>
            <div className="p-2 bg-slate-50 rounded border border-slate-100 flex justify-between">
              <span>Appearance:</span> <span>0.20</span>
            </div>
            <div className="p-2 bg-slate-50 rounded border border-slate-100 flex justify-between">
              <span>Texture (Tiles):</span> <span>0.15</span>
            </div>
            <div className="p-2 bg-slate-50 rounded border border-slate-100 flex justify-between">
              <span>3D Geometry:</span> <span>0.15</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
