import React, { useState, useEffect, Component } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline, Box, Typography, Button } from '@mui/material';

class ErrorBoundary extends Component<{ children: React.ReactNode }, { error: Error | null; didAutoRecover: boolean }> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { error: null, didAutoRecover: false };
  }
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  componentDidCatch(error: Error) {
    // On first crash: clear localStorage and auto-reload once.
    // Use sessionStorage flag so we only attempt auto-recovery once per browser session.
    const alreadyTried = sessionStorage.getItem('_errRecovery') === '1';
    if (!alreadyTried) {
      sessionStorage.setItem('_errRecovery', '1');
      try { localStorage.clear(); } catch {}
      window.location.href = '/';
    }
    // If we get here, auto-recovery didn't help — show the error UI.
    this.setState({ didAutoRecover: true });
  }
  render() {
    if (this.state.error && this.state.didAutoRecover) {
      return (
        <Box sx={{ p: 4, textAlign: 'center', mt: 8 }}>
          <Typography variant="h5" color="error" gutterBottom>Something went wrong</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            The app encountered an unexpected error. Your session data has been reset.
          </Typography>
          <Button variant="contained" onClick={() => { sessionStorage.removeItem('_errRecovery'); localStorage.clear(); window.location.href = '/'; }}>
            Reload App
          </Button>
        </Box>
      );
    }
    return this.props.children;
  }
}

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
  const [sessionId, setSessionId] = useState<string | null>(() => {
    try { return localStorage.getItem('sessionId'); } catch { return null; }
  });
  // Tracks whether processed data exists — lifted so Preprocessing can set it after a successful run
  const [hasProcessed, setHasProcessed] = useState<boolean>(() => {
    try { return localStorage.getItem('hasProcessed') === 'true'; } catch { return false; }
  });

  // App loaded successfully — clear the auto-recovery flag so a future crash
  // can self-heal again (only blocks a second crash-loop, not first recovery).
  useEffect(() => {
    sessionStorage.removeItem('_errRecovery');
  }, []);

  useEffect(() => {
    try {
      if (sessionId) localStorage.setItem('sessionId', sessionId);
      else localStorage.removeItem('sessionId');
    } catch {}
  }, [sessionId]);

  useEffect(() => {
    try { localStorage.setItem('hasProcessed', String(hasProcessed)); } catch {}
  }, [hasProcessed]);

  const handleSessionCreated = (id: string) => {
    setSessionId(id);
    setHasProcessed(false); // new dataset → no processed data yet
  };

  return (
    <ErrorBoundary>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <Routes>
          <Route element={<Layout sessionId={sessionId} />}>
            <Route path="/" element={<Dashboard sessionId={sessionId} />} />
            <Route path="/upload" element={<Upload onSessionCreated={handleSessionCreated} />} />
            <Route
              path="/preview"
              element={sessionId ? <DataPreview sessionId={sessionId} hasProcessed={hasProcessed} /> : <NoSession />}
            />
            <Route
              path="/preprocess"
              element={sessionId ? <Preprocessing sessionId={sessionId} onProcessed={() => setHasProcessed(true)} /> : <NoSession />}
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
    </ErrorBoundary>
  );
}
