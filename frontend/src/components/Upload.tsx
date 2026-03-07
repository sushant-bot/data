import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Paper,
  Typography,
  LinearProgress,
  Alert,
  Chip,
  Card,
  CardContent,
  Grid,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  CheckCircle as SuccessIcon,
  Warning as WarningIcon,
  Storage as RowIcon,
  ViewColumn as ColIcon,
} from '@mui/icons-material';
import { uploadFile } from '../services/api';
import type { UploadResponse } from '../types';

interface UploadProps {
  onSessionCreated: (sessionId: string) => void;
}

export default function Upload({ onSessionCreated }: UploadProps) {
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(
    async (accepted: File[]) => {
      const file = accepted[0];
      if (!file) return;

      if (!file.name.endsWith('.csv')) {
        setError('Only CSV files are supported.');
        return;
      }

      setUploading(true);
      setError(null);
      setResult(null);
      try {
        const res = await uploadFile(file);
        setResult(res);
        onSessionCreated(res.session_id);
      } catch (err: any) {
        setError(err?.response?.data?.error || err.message || 'Upload failed');
      } finally {
        setUploading(false);
      }
    },
    [onSessionCreated]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    maxFiles: 1,
    disabled: uploading,
  });

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto' }}>
      {/* Drop zone */}
      <Paper
        {...getRootProps()}
        elevation={0}
        sx={{
          p: 6,
          textAlign: 'center',
          border: '2px dashed',
          borderColor: isDragActive ? '#6366f1' : '#d1d5db',
          borderRadius: 3,
          bgcolor: isDragActive ? '#eef2ff' : '#fff',
          cursor: uploading ? 'not-allowed' : 'pointer',
          transition: 'all 0.2s',
          '&:hover': { borderColor: '#6366f1', bgcolor: '#f5f3ff' },
        }}
      >
        <input {...getInputProps()} />
        <UploadIcon sx={{ fontSize: 56, color: '#6366f1', mb: 2 }} />
        <Typography variant="h6" fontWeight={600} gutterBottom>
          {isDragActive ? 'Drop your CSV file here' : 'Drag & drop a CSV file'}
        </Typography>
        <Typography color="text.secondary" variant="body2">
          or click to browse — supports .csv files
        </Typography>
      </Paper>

      {uploading && (
        <Box sx={{ mt: 3 }}>
          <LinearProgress sx={{ borderRadius: 1 }} />
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1, textAlign: 'center' }}>
            Uploading and analyzing your dataset...
          </Typography>
        </Box>
      )}

      {error && (
        <Alert severity="error" sx={{ mt: 3, borderRadius: 2 }}>
          {error}
        </Alert>
      )}

      {result && (
        <Box sx={{ mt: 3 }}>
          <Alert
            icon={<SuccessIcon />}
            severity="success"
            sx={{ mb: 3, borderRadius: 2 }}
          >
            Dataset "<strong>{result.dataset_name}</strong>" uploaded successfully!
          </Alert>

          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Card elevation={0} sx={{ border: '1px solid #e5e7eb', borderRadius: 2 }}>
                <CardContent sx={{ textAlign: 'center' }}>
                  <RowIcon sx={{ color: '#6366f1', mb: 1 }} />
                  <Typography variant="h4" fontWeight={700}>
                    {result.statistics.row_count.toLocaleString()}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Rows
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={6}>
              <Card elevation={0} sx={{ border: '1px solid #e5e7eb', borderRadius: 2 }}>
                <CardContent sx={{ textAlign: 'center' }}>
                  <ColIcon sx={{ color: '#6366f1', mb: 1 }} />
                  <Typography variant="h4" fontWeight={700}>
                    {result.statistics.column_count}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Columns
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* PII Warning */}
          {result.pii_detection.pii_detected && (
            <Alert severity="warning" icon={<WarningIcon />} sx={{ mt: 2, borderRadius: 2 }}>
              <Typography fontWeight={600} gutterBottom>
                PII Detected
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                {Object.entries(result.pii_detection.pii_details).map(([col, types]) => (
                  <Chip
                    key={col}
                    label={`${col}: ${(types as string[]).join(', ')}`}
                    size="small"
                    color="warning"
                    variant="outlined"
                  />
                ))}
              </Box>
            </Alert>
          )}

          {/* Data types */}
          <Paper elevation={0} sx={{ mt: 2, p: 2, border: '1px solid #e5e7eb', borderRadius: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Column Data Types
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
              {Object.entries(result.statistics.data_types).map(([col, dtype]) => (
                <Chip key={col} label={`${col}: ${dtype}`} size="small" variant="outlined" />
              ))}
            </Box>
          </Paper>
        </Box>
      )}
    </Box>
  );
}
