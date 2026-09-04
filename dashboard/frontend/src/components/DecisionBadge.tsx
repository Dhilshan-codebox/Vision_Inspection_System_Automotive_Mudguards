import React from 'react';
import { Decision } from '../types/inspection';

interface DecisionBadgeProps {
  decision: Decision | string;
}

export const DecisionBadge: React.FC<DecisionBadgeProps> = ({ decision }) => {
  const normalized = String(decision).toUpperCase();

  let bgColor = 'bg-slate-200 text-slate-800 border-slate-300';
  let icon = '•';

  if (normalized === 'PASS') {
    bgColor = 'bg-emerald-100 text-emerald-800 border-emerald-300';
    icon = '✓';
  } else if (normalized === 'REVIEW') {
    bgColor = 'bg-amber-100 text-amber-800 border-amber-300';
    icon = '⚠';
  } else if (normalized === 'FAIL') {
    bgColor = 'bg-rose-100 text-rose-800 border-rose-300';
    icon = '✕';
  } else if (normalized === 'REQUEST_RECAPTURE') {
    bgColor = 'bg-indigo-100 text-indigo-800 border-indigo-300';
    icon = '⟳';
  }

  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-bold uppercase tracking-wider border shadow-sm ${bgColor}`}>
      <span className="font-mono text-sm">{icon}</span>
      <span>{normalized}</span>
    </span>
  );
};
