import React from 'react';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  status?: 'good' | 'warning' | 'danger' | 'neutral';
}

export const MetricCard: React.FC<MetricCardProps> = ({ title, value, subtitle, status = 'neutral' }) => {
  let statusClass = 'border-slate-200 text-slate-900';
  if (status === 'good') statusClass = 'border-emerald-300 text-emerald-900 bg-emerald-50/30';
  if (status === 'warning') statusClass = 'border-amber-300 text-amber-900 bg-amber-50/30';
  if (status === 'danger') statusClass = 'border-rose-300 text-rose-900 bg-rose-50/30';

  return (
    <div className={`bg-white p-4 rounded-lg border ${statusClass} shadow-sm`}>
      <div className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">{title}</div>
      <div className="text-2xl font-black text-slate-900 my-1">{value}</div>
      {subtitle && <div className="text-xs text-slate-500 font-medium">{subtitle}</div>}
    </div>
  );
};
