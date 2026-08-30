import React from 'react';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  status?: 'good' | 'warning' | 'danger' | 'neutral';
}

export const MetricCard: React.FC<MetricCardProps> = ({ title, value, subtitle, status = 'neutral' }) => {
  let borderColor = 'border-slate-700';
  if (status === 'good') borderColor = 'border-emerald-500';
  if (status === 'warning') borderColor = 'border-amber-500';
  if (status === 'danger') borderColor = 'border-rose-500';

  return (
    <div className={`bg-slate-800 p-4 rounded-lg border ${borderColor} shadow-sm`}>
      <div className="text-xs text-slate-400 font-medium uppercase tracking-wider">{title}</div>
      <div className="text-2xl font-bold text-slate-100 my-1">{value}</div>
      {subtitle && <div className="text-xs text-slate-400">{subtitle}</div>}
    </div>
  );
};
