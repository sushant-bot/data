import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Paper,
  Stepper,
  Step,
  StepLabel,
  Button,
  Skeleton,
  Alert,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  TableChart as PreviewIcon,
  Tune as PreprocessIcon,
  Assessment as QualityIcon,
  ModelTraining as TrainIcon,
  BarChart as VizIcon,
  Psychology as AIIcon,
  ArrowForward as ArrowIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { getSession } from '../services/api';
import type { Session } from '../types';

interface DashboardProps {
  sessionId: string | null;
}

const workflowSteps = [
  { label: 'Upload', key: 'upload_completed', path: '/upload', icon: <UploadIcon /> },
  { label: 'Preview', key: 'preprocessing_completed', path: '/preview', icon: <PreviewIcon /> },
  { label: 'Preprocess', key: 'preprocessing_completed', path: '/preprocess', icon: <PreprocessIcon /> },
  { label: 'Quality', key: 'quality_assessed', path: '/quality', icon: <QualityIcon /> },
  { label: 'Train', key: 'model_trained', path: '/train', icon: <TrainIcon /> },
  { label: 'Visualize', key: 'model_trained', path: '/visualize', icon: <VizIcon /> },
  { label: 'AI Insights', key: 'recommendations_generated', path: '/recommendations', icon: <AIIcon /> },
];

const quickActions = [
  { label: 'Upload Dataset', description: 'Upload a CSV file to get started', path: '/upload', icon: <UploadIcon sx={{ fontSize: 40, color: '#6366f1' }} />, color: '#eef2ff' },
  { label: 'View Data', description: 'Preview and explore your data', path: '/preview', icon: <PreviewIcon sx={{ fontSize: 40, color: '#0891b2' }} />, color: '#ecfeff' },
  { label: 'Train Model', description: 'Build ML models on your data', path: '/train', icon: <TrainIcon sx={{ fontSize: 40, color: '#7c3aed' }} />, color: '#f5f3ff' },
  { label: 'Get AI Insights', description: 'AI-powered recommendations', path: '/recommendations', icon: <AIIcon sx={{ fontSize: 40, color: '#059669' }} />, color: '#f0fdf4' },
];

export default function Dashboard({ sessionId }: DashboardProps) {
  const navigate = useNavigate();
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!sessionId) return;
    setLoading(true);
    getSession(sessionId)
      .then(setSession)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [sessionId]);

  const activeStep = session
    ? workflowSteps.filter((s) => session.workflow_state[s.key as keyof typeof session.workflow_state]).length
    : 0;

  return (
    <Box>
      {/* Hero */}
      <Paper
        elevation={0}
        sx={{
          p: 4,
          mb: 3,
          borderRadius: 3,
          background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
          color: '#fff',
        }}
      >
        <Typography variant="h4" fontWeight={700} gutterBottom>
          Welcome to AI Data Analyst
        </Typography>
        <Typography variant="body1" sx={{ opacity: 0.85, maxWidth: 600 }}>
          Upload your CSV dataset and leverage machine learning, AI recommendations, and automated quality assessment to extract powerful insights.
        </Typography>
        {!sessionId && (
          <Button
            variant="contained"
            onClick={() => navigate('/upload')}
            endIcon={<ArrowIcon />}
            sx={{
              mt: 2,
              bgcolor: 'rgba(255,255,255,0.2)',
              '&:hover': { bgcolor: 'rgba(255,255,255,0.3)' },
              textTransform: 'none',
              borderRadius: 2,
            }}
          >
            Get Started
          </Button>
        )}
      </Paper>

      {/* Workflow stepper */}
      {sessionId && (
        <Paper elevation={0} sx={{ p: 3, mb: 3, border: '1px solid #e5e7eb', borderRadius: 2 }}>
          <Typography variant="subtitle1" fontWeight={600} gutterBottom>
            Workflow Progress
          </Typography>
          {loading ? (
            <Skeleton variant="rectangular" height={60} sx={{ borderRadius: 2 }} />
          ) : (
            <Stepper activeStep={activeStep} alternativeLabel>
              {workflowSteps.map((step) => (
                <Step
                  key={step.label}
                  onClick={() => navigate(step.path)}
                  sx={{ cursor: 'pointer' }}
                >
                  <StepLabel>{step.label}</StepLabel>
                </Step>
              ))}
            </Stepper>
          )}
          {session && (
            <Box sx={{ mt: 2, display: 'flex', gap: 1, alignItems: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                Dataset: <strong>{session.dataset_name}</strong>
              </Typography>
              <Typography variant="body2" color="text.secondary">
                — Status: <strong>{session.status}</strong>
              </Typography>
            </Box>
          )}
        </Paper>
      )}

      {/* Quick actions */}
      <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 2 }}>
        Quick Actions
      </Typography>
      <Grid container spacing={2}>
        {quickActions.map((action) => (
          <Grid item xs={12} sm={6} md={3} key={action.label}>
            <Card
              elevation={0}
              onClick={() => navigate(action.path)}
              sx={{
                border: '1px solid #e5e7eb',
                borderRadius: 2,
                cursor: 'pointer',
                transition: 'all 0.2s',
                '&:hover': { borderColor: '#6366f1', transform: 'translateY(-2px)' },
              }}
            >
              <CardContent sx={{ textAlign: 'center', py: 3 }}>
                <Box sx={{ mb: 1.5, p: 1.5, borderRadius: 2, bgcolor: action.color, display: 'inline-flex' }}>
                  {action.icon}
                </Box>
                <Typography variant="body1" fontWeight={600}>{action.label}</Typography>
                <Typography variant="caption" color="text.secondary">{action.description}</Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}
