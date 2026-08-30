import React from 'react';

interface ReasoningTraceProps {
  inspectionId: string;
  ruleFired?: string;
  reasoningChain: string[];
  modelVersion?: string;
}

export const ReasoningTrace: React.FC<ReasoningTraceProps> = ({
  inspectionId,
  ruleFired,
  reasoningChain,
  modelVersion = '2.0.0',
}) => {
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg p-4 text-xs font-mono space-y-3">
      <div className="flex flex-wrap justify-between border-b border-slate-700 pb-2 text-slate-300">
        <div>
          <span className="text-slate-500">Inspection ID: </span>
          <span className="text-emerald-400 font-bold">{inspectionId}</span>
        </div>
        <div>
          <span className="text-slate-500">Model Version: </span>
          <span className="text-slate-200">{modelVersion}</span>
        </div>
      </div>

      <div>
        <div className="text-slate-400 font-semibold mb-1">Fired Decision Rule:</div>
        <div className="bg-slate-900 px-3 py-1.5 rounded text-amber-300 border border-slate-800">
          {ruleFired || 'DEFAULT_SAFETY_FALLBACK'}
        </div>
      </div>

      <div>
        <div className="text-slate-400 font-semibold mb-1">Reasoning Chain:</div>
        {reasoningChain && reasoningChain.length > 0 ? (
          <ul className="list-disc list-inside space-y-1 bg-slate-900 p-2.5 rounded text-slate-200">
            {reasoningChain.map((reason, idx) => (
              <li key={idx}>{reason}</li>
            ))}
          </ul>
        ) : (
          <p className="text-slate-500 italic">No reasoning chain supplied.</p>
        )}
      </div>
    </div>
  );
};
