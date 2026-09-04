import React from 'react';
import { MetricCard } from '../components/MetricCard';

export const EvaluationPage: React.FC = () => {
  return (
    <div className="space-y-6">
      <div className="bg-white p-4 rounded-lg border border-[#dbe2ea] shadow-sm flex justify-between items-center">
        <div>
          <h1 className="text-xl font-bold text-slate-900">Evaluation & Error Analysis</h1>
          <p className="text-xs text-slate-500 mt-0.5">Model precision, recall, F1 scores, and error taxonomy gallery</p>
        </div>
        <span className="font-mono text-xs text-slate-600 bg-slate-100 px-3 py-1 rounded font-bold border border-slate-200">
          Validation Split: val_set_v1 (240 samples)
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MetricCard title="Macro F1 Score" value="0.942" subtitle="+2.1% vs baseline" status="good" />
        <MetricCard title="Mean Precision" value="0.951" subtitle="Low false-fail rate" status="good" />
        <MetricCard title="Mean Recall" value="0.934" subtitle="High defect catch rate" status="good" />
        <MetricCard title="Calibration ECE" value="0.038" subtitle="Expected Calibration Error" status="good" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-4 rounded-lg border border-[#dbe2ea] shadow-sm space-y-3">
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-600">Per-Class Metrics</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-2">Class</th>
                  <th className="py-2">Precision</th>
                  <th className="py-2">Recall</th>
                  <th className="py-2">F1 Score</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                <tr><td className="py-2 font-sans font-semibold">Scratch</td><td>0.94</td><td>0.92</td><td>0.93</td></tr>
                <tr><td className="py-2 font-sans font-semibold">Dent</td><td>0.96</td><td>0.95</td><td>0.95</td></tr>
                <tr><td className="py-2 font-sans font-semibold">Paint Defect</td><td>0.92</td><td>0.90</td><td>0.91</td></tr>
                <tr><td className="py-2 font-sans font-semibold">Paint Misalignment</td><td>0.95</td><td>0.93</td><td>0.94</td></tr>
                <tr><td className="py-2 font-sans font-semibold">Good (Nominal)</td><td>0.98</td><td>0.97</td><td>0.97</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg border border-[#dbe2ea] shadow-sm space-y-3">
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-600">Confusion Matrix (Validation Set)</h2>
          <div className="grid grid-cols-5 gap-1 text-center font-mono text-[11px] p-2 bg-slate-50 rounded border border-slate-200">
            <div className="bg-emerald-100 font-bold p-2 rounded text-emerald-900">48 (Scratch)</div>
            <div className="bg-slate-100 p-2 rounded text-slate-600">2</div>
            <div className="bg-slate-100 p-2 rounded text-slate-600">0</div>
            <div className="bg-slate-100 p-2 rounded text-slate-600">1</div>
            <div className="bg-slate-100 p-2 rounded text-slate-600">0</div>

            <div className="bg-slate-100 p-2 rounded text-slate-600">1</div>
            <div className="bg-emerald-100 font-bold p-2 rounded text-emerald-900">45 (Dent)</div>
            <div className="bg-slate-100 p-2 rounded text-slate-600">1</div>
            <div className="bg-slate-100 p-2 rounded text-slate-600">0</div>
            <div className="bg-slate-100 p-2 rounded text-slate-600">0</div>

            <div className="bg-slate-100 p-2 rounded text-slate-600">0</div>
            <div className="bg-slate-100 p-2 rounded text-slate-600">0</div>
            <div className="bg-emerald-100 font-bold p-2 rounded text-emerald-900">38 (Paint)</div>
            <div className="bg-slate-100 p-2 rounded text-slate-600">1</div>
            <div className="bg-slate-100 p-2 rounded text-slate-600">1</div>

            <div className="bg-slate-100 p-2 rounded text-slate-600">0</div>
            <div className="bg-slate-100 p-2 rounded text-slate-600">0</div>
            <div className="bg-slate-100 p-2 rounded text-slate-600">1</div>
            <div className="bg-emerald-100 font-bold p-2 rounded text-emerald-900">30 (Align)</div>
            <div className="bg-slate-100 p-2 rounded text-slate-600">0</div>

            <div className="bg-slate-100 p-2 rounded text-slate-600">1</div>
            <div className="bg-slate-100 p-2 rounded text-slate-600">0</div>
            <div className="bg-slate-100 p-2 rounded text-slate-600">0</div>
            <div className="bg-slate-100 p-2 rounded text-slate-600">0</div>
            <div className="bg-emerald-100 font-bold p-2 rounded text-emerald-900">71 (Good)</div>
          </div>
        </div>
      </div>
    </div>
  );
};
