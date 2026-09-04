import React, { useState } from 'react';
import { AppShell } from './components/AppShell';
import { InspectionPage } from './pages/Inspection';
import { DatasetAuditPage } from './pages/DatasetAudit';
import { EvaluationPage } from './pages/Evaluation';
import { ModelsPage } from './pages/Models';
import { MonitoringPage } from './pages/Monitoring';
import { SettingsPage } from './pages/Settings';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('inspection');

  const renderContent = () => {
    switch (activeTab) {
      case 'inspection':
        return <InspectionPage />;
      case 'audit':
        return <DatasetAuditPage />;
      case 'evaluation':
        return <EvaluationPage />;
      case 'models':
        return <ModelsPage />;
      case 'monitoring':
        return <MonitoringPage />;
      case 'settings':
        return <SettingsPage />;
      default:
        return <InspectionPage />;
    }
  };

  return (
    <AppShell activeTab={activeTab} onTabChange={setActiveTab}>
      {renderContent()}
    </AppShell>
  );
};

export default App;
