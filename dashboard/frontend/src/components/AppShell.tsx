import React from 'react';

interface AppShellProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
  children: React.ReactNode;
}

export const AppShell: React.FC<AppShellProps> = ({ activeTab, onTabChange, children }) => {
  const tabs = [
    { id: 'inspection', label: 'Inspection Workspace' },
    { id: 'audit', label: 'Dataset Audit' },
    { id: 'evaluation', label: 'Evaluation & Error Analysis' },
    { id: 'models', label: 'Model Registry' },
    { id: 'monitoring', label: 'Edge Monitoring' },
    { id: 'settings', label: 'System Settings' },
  ];

  return (
    <div className="min-h-screen bg-[#f7f9fc] text-[#172033] flex flex-col font-sans">
      {/* Top Bar */}
      <header className="h-14 bg-white border-b border-[#dbe2ea] px-6 flex items-center justify-between shadow-sm sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <span className="font-black tracking-tight text-lg text-slate-900">Mudguard AI</span>
          <span className="text-slate-300">|</span>
          <span className="bg-[#eaf0ff] text-[#2457d6] text-[11px] font-bold px-2.5 py-0.5 rounded border border-[#2457d6]/20">
            SIMULATION MODE
          </span>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono text-slate-600">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
            <span>Pipeline: Online</span>
          </div>
          <span>Active Model: v2.0.0</span>
          <span>Config: v1.0.0</span>
        </div>
      </header>

      {/* Main Workstation Layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Navigation Sidebar */}
        <aside className="w-60 bg-white border-r border-[#dbe2ea] p-4 flex flex-col gap-1 shadow-sm">
          <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider px-3 mb-2">Navigation</div>
          {tabs.map((t) => {
            const isActive = activeTab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => onTabChange(t.id)}
                className={`w-full text-left px-3 py-2 rounded text-xs font-semibold transition-colors ${
                  isActive
                    ? 'bg-[#2457d6] text-white shadow-sm'
                    : 'text-slate-700 hover:bg-[#f1f4f8]'
                }`}
              >
                {t.label}
              </button>
            );
          })}
        </aside>

        {/* Content Area */}
        <main className="flex-1 overflow-auto p-6 max-w-7xl">
          {children}
        </main>
      </div>
    </div>
  );
};
