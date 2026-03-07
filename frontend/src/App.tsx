import React, { useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';

import Layout from './components/Layout';
import Dashboard from './components/Dashboard';
import Upload from './components/Upload';
import DataPreview from './components/DataPreview';
import Preprocessing from './components/Preprocessing';
import QualityAssessment from './components/QualityAssessment';
import MLTraining from './components/MLTraining';
import Visualizations from './components/Visualizations';
import AIRecommendations from './components/AIRecommendations';

const theme = createTheme({
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
  },
  palette: {
    primary: { main: '#6366f1' },
    background: { default: '#f5f7fa' },
  },
  shape: { borderRadius: 8 },
});

function NoSession() {
  return (
    <div style={{ textAlign: 'center', padding: '60px 20px', color: '#6b7280' }}>
      <h2 style={{ color: '#111827', marginBottom: 8 }}>No Active Session</h2>
      <p>Upload a dataset first to access this feature.</p>
    </div>
  );
}

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <Routes>
          <Route element={<Layout sessionId={sessionId} />}>
            <Route path="/" element={<Dashboard sessionId={sessionId} />} />
            <Route path="/upload" element={<Upload onSessionCreated={setSessionId} />} />
            <Route
              path="/preview"
              element={sessionId ? <DataPreview sessionId={sessionId} /> : <NoSession />}
            />
            <Route
              path="/preprocess"
              element={sessionId ? <Preprocessing sessionId={sessionId} /> : <NoSession />}
            />
            <Route
              path="/quality"
              element={sessionId ? <QualityAssessment sessionId={sessionId} /> : <NoSession />}
            />
            <Route
              path="/train"
              element={sessionId ? <MLTraining sessionId={sessionId} /> : <NoSession />}
            />
            <Route
              path="/visualize"
              element={sessionId ? <Visualizations sessionId={sessionId} /> : <NoSession />}
            />
            <Route
              path="/recommendations"
              element={sessionId ? <AIRecommendations sessionId={sessionId} /> : <NoSession />}
            />
          </Route>
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}
