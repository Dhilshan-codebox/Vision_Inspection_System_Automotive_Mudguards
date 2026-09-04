import React from 'react';

interface ReasoningTraceProps {
  inspectionId: string;
  ruleFired?: string;
  reasoningChain?: string[];
  modelVersion?: string;
  configVersion?: string;
  calibrationVersion?: string;
}

export const ReasoningTrace: React.FC<ReasoningTraceProps> = ({
  inspectionId,
  ruleFired,
  reasoningChain = [],
  modelVersion = '2.0.0',
  configVersion = '1.0.0',
  calibrationVersion = '1.0.0',
}) => {
  return (
    <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm text-xs space-y-3">
      <div className="flex flex-wrap items-center justify-between border-b border-slate-100 pb-2 text-slate-600 font-mono">
        <div>
          <span className="text-slate-400">ID: </span>
          <span className="font-bold text-slate-900">{inspectionId}</span>
        </div>
        <div className="flex gap-2 text-[11px] text-slate-500">
          <span>Model v{modelVersion}</span>
          <span>•</span>
          <span>Config v{configVersion}</span>
          <span>•</span>
          <span>Cal v{calibrationVersion}</span>
        </div>
      </div>

      <div>
        <div className="text-slate-500 font-semibold mb-1">Fired Decision Rule:</div>
        <div className="bg-slate-50 font-mono text-slate-900 px-3 py-1.5 rounded border border-slate-200 font-bold">
          {ruleFired || 'DEFAULT_SAFETY_FALLBACK'}
        </div>
      </div>

      <div>
        <div className="text-slate-500 font-semibold mb-1">Evidence Reasoning Chain:</div>
        {reasoningChain && reasoningChain.length > 0 ? (
          <ul className="list-disc list-inside space-y-1 bg-slate-50 p-2.5 rounded border border-slate-200 text-slate-800 font-sans">
            {reasoningChain.map((reason, idx) => (
              <li key={idx}>{reason}</li>
            ))}
          </ul>
        ) : (
          <p className="text-slate-400 italic">No reasoning chain supplied.</p>
        )}
      </div>
    </div>
  );
};
