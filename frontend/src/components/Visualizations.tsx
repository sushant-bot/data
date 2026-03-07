import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  Grid,
  LinearProgress,
  Card,
  CardContent,
  CardMedia,
} from '@mui/material';
import {
  BarChart as ChartIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { generateVisualization } from '../services/api';
import type { VisualizationType, VisualizationResponse } from '../types';

interface VisualizationsProps {
  sessionId: string;
}

const vizOptions: { value: VisualizationType; label: string; description: string }[] = [
  { value: 'correlation_heatmap', label: 'Correlation Heatmap', description: 'Show correlations between numeric features' },
  { value: 'confusion_matrix', label: 'Confusion Matrix', description: 'Visualize classification model predictions' },
  { value: 'roc_curve', label: 'ROC Curve', description: 'Receiver Operating Characteristic curve' },
  { value: 'cluster_plot', label: 'Cluster Plot', description: 'Visualize clustering results' },
  { value: 'feature_importance', label: 'Feature Importance', description: 'Show most influential features' },
];

export default function Visualizations({ sessionId }: VisualizationsProps) {
  const [vizType, setVizType] = useState<VisualizationType>('correlation_heatmap');
  const [datasetType, setDatasetType] = useState<'original' | 'processed'>('original');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<VisualizationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await generateVisualization({
        session_id: sessionId,
        visualization_type: vizType,
        parameters: { dataset_type: datasetType },
      });
      setResult(res);
    } catch (err: any) {
      setError(err?.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ maxWidth: 900, mx: 'auto' }}>
      {/* Viz type selector */}
      <Paper elevation={0} sx={{ p: 3, mb: 3, border: '1px solid #e5e7eb', borderRadius: 2 }}>
        <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 2 }}>
          Choose Visualization
        </Typography>
        <Grid container spacing={2}>
          {vizOptions.map((v) => (
            <Grid item xs={6} md={4} key={v.value}>
              <Card
                elevation={0}
                onClick={() => setVizType(v.value)}
                sx={{
                  cursor: 'pointer',
                  border: vizType === v.value ? '2px solid #6366f1' : '1px solid #e5e7eb',
                  borderRadius: 2,
                  bgcolor: vizType === v.value ? '#eef2ff' : '#fff',
                  transition: 'all 0.15s',
                  '&:hover': { borderColor: '#6366f1' },
                }}
              >
                <CardContent sx={{ py: 2 }}>
                  <Typography variant="body2" fontWeight={600}>{v.label}</Typography>
                  <Typography variant="caption" color="text.secondary">{v.description}</Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>

        <Box sx={{ display: 'flex', gap: 2, mt: 3, alignItems: 'center' }}>
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel>Dataset</InputLabel>
            <Select
              value={datasetType}
              label="Dataset"
              onChange={(e) => setDatasetType(e.target.value as 'original' | 'processed')}
            >
              <MenuItem value="original">Original</MenuItem>
              <MenuItem value="processed">Processed</MenuItem>
            </Select>
          </FormControl>
          <Button
            variant="contained"
            startIcon={<ChartIcon />}
            onClick={handleGenerate}
            disabled={loading}
            sx={{
              borderRadius: 2,
              textTransform: 'none',
              bgcolor: '#6366f1',
              '&:hover': { bgcolor: '#4f46e5' },
            }}
          >
            Generate
          </Button>
        </Box>
      </Paper>

      {loading && <LinearProgress sx={{ borderRadius: 1, mb: 2 }} />}
      {error && <Alert severity="error" sx={{ borderRadius: 2, mb: 2 }}>{error}</Alert>}

      {/* Result Image */}
      {result && (
        <Paper elevation={0} sx={{ border: '1px solid #e5e7eb', borderRadius: 2, overflow: 'hidden' }}>
          <Box sx={{ p: 2, borderBottom: '1px solid #e5e7eb', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="subtitle1" fontWeight={600}>
              {vizOptions.find((v) => v.value === result.visualization_type)?.label}
            </Typography>
            <Button
              size="small"
              startIcon={<RefreshIcon />}
              onClick={handleGenerate}
              sx={{ textTransform: 'none' }}
            >
              Regenerate
            </Button>
          </Box>
          <Box sx={{ p: 2, textAlign: 'center' }}>
            <img
              src={result.presigned_url}
              alt={result.visualization_type}
              style={{ maxWidth: '100%', borderRadius: 8 }}
            />
          </Box>
        </Paper>
      )}
    </Box>
  );
}
