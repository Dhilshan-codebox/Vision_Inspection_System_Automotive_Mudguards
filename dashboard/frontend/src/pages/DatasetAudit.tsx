import React from 'react';
import { MetricCard } from '../components/MetricCard';

export const DatasetAuditPage: React.FC = () => {
  return (
    <div className="space-y-6">
      <div className="bg-white p-4 rounded-lg border border-[#dbe2ea] shadow-sm flex justify-between items-center">
        <div>
          <h1 className="text-xl font-bold text-slate-900">Dataset Audit & Integrity Engine</h1>
          <p className="text-xs text-slate-500 mt-0.5">Non-destructive dataset discovery, SHA-256 duplicate tracking, and split-leakage verification</p>
        </div>
        <span className="bg-emerald-100 text-emerald-800 border border-emerald-300 text-xs font-bold px-3 py-1 rounded">
          QUALITY GATE: PASS
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MetricCard title="Total Discoveries" value="1,240" subtitle="Images indexed across splits" status="neutral" />
        <MetricCard title="Valid Images" value="1,240" subtitle="100% readable format" status="good" />
        <MetricCard title="Corrupted Files" value="0" subtitle="Zero corrupted headers" status="good" />
        <MetricCard title="Split Leakage" value="0" subtitle="Zero cross-split duplicate hashes" status="good" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-4 rounded-lg border border-[#dbe2ea] shadow-sm space-y-3">
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-600">Defect Class Breakdown</h2>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between items-center p-2 bg-slate-50 rounded border border-slate-100 font-medium">
              <span>Scratch</span>
              <span className="font-mono text-slate-800 font-bold">320 images (25.8%)</span>
            </div>
            <div className="flex justify-between items-center p-2 bg-slate-50 rounded border border-slate-100 font-medium">
              <span>Dent</span>
              <span className="font-mono text-slate-800 font-bold">280 images (22.5%)</span>
            </div>
            <div className="flex justify-between items-center p-2 bg-slate-50 rounded border border-slate-100 font-medium">
              <span>Paint Defect</span>
              <span className="font-mono text-slate-800 font-bold">190 images (15.3%)</span>
            </div>
            <div className="flex justify-between items-center p-2 bg-slate-50 rounded border border-slate-100 font-medium">
              <span>Paint Misalignment</span>
              <span className="font-mono text-slate-800 font-bold">150 images (12.1%)</span>
            </div>
            <div className="flex justify-between items-center p-2 bg-slate-50 rounded border border-slate-100 font-medium">
              <span>Good (Nominal)</span>
              <span className="font-mono text-slate-800 font-bold">300 images (24.2%)</span>
            </div>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg border border-[#dbe2ea] shadow-sm space-y-3">
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-600">Audit Findings & Verification Log</h2>
          <div className="bg-slate-50 p-3 rounded border border-slate-200 text-xs font-mono space-y-2 text-slate-700">
            <div className="text-emerald-700 font-bold">[INFO] SHA-256 duplicate verification completed in 1.2s</div>
            <div>[INFO] Checked 1,240 image files across train/val/test splits.</div>
            <div>[INFO] Resolution dimensions verified: 640x640 min, 1920x1080 max.</div>
            <div className="text-slate-500 italic">No critical findings or quality gate violations detected.</div>
          </div>
        </div>
      </div>
    </div>
  );
};
