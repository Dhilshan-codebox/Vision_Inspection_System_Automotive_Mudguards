import React from 'react';
import { Evidence, Prediction } from '../types/inspection';

interface EvidenceViewerProps {
  imageUrl?: string;
  prediction?: Prediction;
  evidence: Evidence[];
}

export const EvidenceViewer: React.FC<EvidenceViewerProps> = ({ imageUrl, prediction, evidence }) => {
  return (
    <div className="relative border border-slate-700 rounded-lg overflow-hidden bg-slate-900 min-h-[360px] flex flex-col items-center justify-center p-4">
      {imageUrl ? (
        <div className="relative w-full max-w-xl">
          <img src={imageUrl} alt="Mudguard Inspection" className="w-full h-auto rounded shadow-md" />
          {prediction && prediction.bbox && (
            <div
              className="absolute border-2 border-red-500 bg-red-500/20 rounded pointer-events-none"
              style={{
                left: `${(prediction.bbox[0] / 640) * 100}%`,
                top: `${(prediction.bbox[1] / 640) * 100}%`,
                width: `${((prediction.bbox[2] - prediction.bbox[0]) / 640) * 100}%`,
                height: `${((prediction.bbox[3] - prediction.bbox[1]) / 640) * 100}%`,
              }}
            >
              <span className="absolute -top-6 left-0 bg-red-600 text-white text-xs px-1.5 py-0.5 rounded font-mono">
                {prediction.label} ({(prediction.confidence * 100).toFixed(1)}%)
              </span>
            </div>
          )}
        </div>
      ) : (
        <div className="text-slate-400 text-center">
          <p className="font-semibold text-base mb-1">No Image Loaded</p>
          <p className="text-xs">Upload a mudguard image to trigger inspection analysis</p>
        </div>
      )}

      <div className="w-full mt-4 grid grid-cols-2 md:grid-cols-4 gap-2">
        {evidence.map((ev) => (
          <div key={ev.perspective} className="bg-slate-800 p-2.5 rounded text-xs border border-slate-700">
            <div className="flex justify-between font-medium capitalize text-slate-200">
              <span>{ev.perspective}</span>
              <span className={ev.status === 'AVAILABLE' ? 'text-emerald-400' : 'text-amber-400'}>{ev.status}</span>
            </div>
            <div className="mt-1 text-slate-400">
              Score: <span className="font-mono text-slate-100">{ev.score !== undefined && ev.score !== null ? ev.score.toFixed(3) : 'N/A'}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
