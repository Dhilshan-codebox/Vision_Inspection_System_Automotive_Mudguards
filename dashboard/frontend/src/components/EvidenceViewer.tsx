import React, { useState } from 'react';
import { Evidence, Prediction } from '../types/inspection';

interface EvidenceViewerProps {
  imageUrl?: string;
  prediction?: Prediction;
  evidence: Evidence[];
}

export const EvidenceViewer: React.FC<EvidenceViewerProps> = ({ imageUrl, prediction, evidence }) => {
  const [zoom, setZoom] = useState<number>(1.0);
  const [showOverlays, setShowOverlays] = useState<boolean>(true);

  const geometryEv = evidence.find((e) => e.perspective === 'geometry');
  const isGeometryUnavailable = !geometryEv || geometryEv.status === 'UNAVAILABLE' || geometryEv.status === 'unavailable';

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 pb-3">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-700">
          <span>Evidence Viewer</span>
          {isGeometryUnavailable && (
            <span className="bg-amber-50 text-amber-800 border border-amber-200 px-2 py-0.5 rounded text-[11px]">
              Geometry unavailable (No 3D sensor)
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 text-xs">
          <label className="flex items-center gap-1.5 cursor-pointer text-slate-600">
            <input
              type="checkbox"
              checked={showOverlays}
              onChange={(e) => setShowOverlays(e.target.checked)}
              className="rounded border-slate-300 text-blue-600"
            />
            <span>Show Overlays</span>
          </label>

          <div className="flex items-center border border-slate-200 rounded overflow-hidden">
            <button
              onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}
              className="px-2 py-0.5 bg-slate-50 hover:bg-slate-100 text-slate-700 border-r border-slate-200"
            >
              -
            </button>

            <span className="px-2 py-0.5 text-slate-600 font-mono text-[11px]">{Math.round(zoom * 100)}%</span>

            <button
              onClick={() => setZoom((z) => Math.min(2.5, z + 0.25))}
              className="px-2 py-0.5 bg-slate-50 hover:bg-slate-100 text-slate-700 border-l border-slate-200"
            >
              +
            </button>

            <button
              onClick={() => setZoom(1.0)}
              className="px-2 py-0.5 bg-slate-50 hover:bg-slate-100 text-slate-700 border-l border-slate-200 text-[11px]"
            >
              Reset
            </button>
          </div>
        </div>
      </div>

      <div className="relative border border-slate-200 rounded bg-slate-900 min-h-[360px] flex items-center justify-center overflow-hidden p-2">
        {imageUrl ? (
          <div className="relative max-w-full overflow-auto" style={{ transform: `scale(${zoom})`, transformOrigin: 'center center' }}>
            <img src={imageUrl} alt="Inspection Capture" className="max-h-[500px] w-auto rounded shadow-sm" />

            {showOverlays && prediction && prediction.bbox && (
              <div
                className="absolute border-2 border-red-500 bg-red-500/20 rounded pointer-events-none"
                style={{
                  left: `${(prediction.bbox[0] / 640) * 100}%`,
                  top: `${(prediction.bbox[1] / 640) * 100}%`,
                  width: `${((prediction.bbox[2] - prediction.bbox[0]) / 640) * 100}%`,
                  height: `${((prediction.bbox[3] - prediction.bbox[1]) / 640) * 100}%`,
                }}
              >
                <span className="absolute -top-6 left-0 bg-red-600 text-white text-[11px] font-bold px-1.5 py-0.5 rounded shadow-sm">
                  {prediction.label} ({(prediction.confidence * 100).toFixed(1)}%)
                </span>
              </div>
            )}
          </div>
        ) : (
          <div className="text-slate-400 text-center p-8">
            <p className="font-semibold text-sm">No Image Loaded</p>
            <p className="text-xs text-slate-500 mt-1">Upload an image or select a predefined simulation sample</p>
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-2 pt-2 border-t border-slate-100">
        {evidence.map((ev) => (
          <div key={ev.perspective} className="bg-slate-50 p-2 rounded border border-slate-200 text-xs">
            <div className="flex justify-between items-center font-medium capitalize text-slate-800">
              <span>{ev.perspective}</span>
              <span className={ev.status === 'AVAILABLE' || ev.status === 'available' ? 'text-emerald-700 font-bold' : 'text-slate-400'}>
                {ev.status}
              </span>
            </div>

            <div className="mt-1 text-slate-600 font-mono text-[11px]">
              Score: {ev.score !== undefined && ev.score !== null ? ev.score.toFixed(3) : 'N/A'}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
