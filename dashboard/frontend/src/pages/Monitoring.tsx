import React from 'react';
import { MetricCard } from '../components/MetricCard';

export const MonitoringPage: React.FC = () => {
  return (
    <div className="space-y-6">
      <div className="bg-white p-4 rounded-lg border border-[#dbe2ea] shadow-sm flex justify-between items-center">
        <div>
          <h1 className="text-xl font-bold text-slate-900">Edge Monitoring & Telemetry</h1>
          <p className="text-xs text-slate-500 mt-0.5">Live inspection throughput, stage latency percentiles, and hardware status</p>
        </div>
        <span className="flex items-center gap-1.5 text-xs font-bold text-emerald-800 bg-emerald-50 border border-emerald-300 px-3 py-1 rounded">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          EDGE NODE: ONLINE
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MetricCard title="Throughput FPS" value="54.0" subtitle="Frames processed / sec" status="good" />
        <MetricCard title="p50 Latency" value="16.2 ms" subtitle="Median pipeline response" status="good" />
        <MetricCard title="p95 Latency" value="24.1 ms" subtitle="95th percentile peak" status="good" />
        <MetricCard title="Image Quality Failures" value="0.4%" subtitle="Blur & exposure rejections" status="good" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-4 rounded-lg border border-[#dbe2ea] shadow-sm space-y-3">
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-600">Pipeline Stage Latency Breakdown</h2>
          <div className="space-y-2 text-xs font-mono">
            <div className="flex justify-between items-center p-2 bg-slate-50 rounded border border-slate-100">
              <span>Stage 0: Image Quality Gate</span>
              <span className="text-slate-800 font-bold">1.2 ms</span>
            </div>
            <div className="flex justify-between items-center p-2 bg-slate-50 rounded border border-slate-100">
              <span>Stage 1: Multi-Head Backbone Screening</span>
              <span className="text-slate-800 font-bold">8.4 ms</span>
            </div>
            <div className="flex justify-between items-center p-2 bg-slate-50 rounded border border-slate-100">
              <span>Stage 2: High-Res Tiled Texture Analysis</span>
              <span className="text-slate-800 font-bold">4.8 ms</span>
            </div>
            <div className="flex justify-between items-center p-2 bg-slate-50 rounded border border-slate-100">
              <span>Stage 3: 3D Geometry Depth Verification</span>
              <span className="text-slate-800 font-bold">1.1 ms</span>
            </div>
            <div className="flex justify-between items-center p-2 bg-slate-50 rounded border border-slate-100">
              <span>Stage 4: Calibrated Late Fusion & Reasoning</span>
              <span className="text-slate-800 font-bold">0.7 ms</span>
            </div>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg border border-[#dbe2ea] shadow-sm space-y-3">
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-600">Hardware Telemetry & Integrations</h2>
          <div className="bg-slate-50 p-6 rounded border border-slate-200 text-center text-slate-500 space-y-2">
            <div className="font-bold text-slate-700 text-xs">SIMULATION MODE ACTIVE</div>
            <p className="text-[11px] max-w-sm mx-auto">
              Hardware integrations (Industrial Camera GigE, PLC Actuator, Robot Arm) are intentionally disabled during software simulation mode.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
