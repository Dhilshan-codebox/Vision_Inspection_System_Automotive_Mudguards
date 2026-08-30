import React from 'react';

interface DecisionBadgeProps {
  decision: 'PASS' | 'REVIEW' | 'FAIL' | 'REQUEST_RECAPTURE' | string;
}

export const DecisionBadge: React.FC<DecisionBadgeProps> = ({ decision }) => {
  let bgColor = 'bg-gray-500';
  let textColor = 'text-white';

  switch (decision) {
    case 'PASS':
      bgColor = 'bg-emerald-600';
      break;
    case 'REVIEW':
      bgColor = 'bg-amber-500';
      break;
    case 'FAIL':
      bgColor = 'bg-rose-600';
      break;
    case 'REQUEST_RECAPTURE':
      bgColor = 'bg-indigo-600';
      break;
  }

  return (
    <span className={`inline-flex items-center px-4 py-1.5 rounded-full text-sm font-bold tracking-wide shadow-sm ${bgColor} ${textColor}`}>
      {decision}
    </span>
  );
};
